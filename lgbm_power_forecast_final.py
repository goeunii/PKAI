"""
최종 제조 전력 예측 + 피크 경보 모델

[베이스라인]
- 기존 56개 입력변수 유지
- P50 / P90 / Peak classifier를 한 파일에서 함께 학습

[최종 채택 파생변수]
1) P50 전용: 생산전환상태
   - 무생산 / 시작 / 증가 / 감소 / 종료 / 유지 상태를 표현
   - 실험에서 최대전력 P50 WAPE와 오전/전환구간 MAE가 함께 개선되어 채택

2) 최대전력 P90 전용: 오전x생산변화
   - 09~12시 생산량 변화량을 직접 표현
   - 최대전력 P90 Pinball 감소 + Coverage가 90%에 가까워져 채택
   - 평균전력 P90에는 개선이 없어서 넣지 않음

3) Peak classifier
   - 기존 56개 baseline 변수 그대로 사용
   - 생산량_차이_168h는 FP를 줄였지만 FN이 2 -> 4로 증가하고 PR-AUC가 하락하여 미채택

[시간 분할]
- 1~6월       : 하이퍼파라미터 / iteration 탐색 학습
- 7월         : 모델 선택
- 1~7월       : 최종 재학습
- 8/1~8/14    : Peak threshold 선택
- 8/15~9/14   : 장기 평가

주의
- 다음 날 시간대별 생산계획을 사전에 알고 있다는 운영 가정이 필요하다.
- 현재 P90 최종값은 Raw P90을 사용한다. 이전 calibration offset은 과도한 coverage를 만들었으므로 제외했다.
"""

from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")


# ============================================================
# 0. 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "okm_augumented_2021_preprocessed.csv",
    BASE_DIR.parent / "data" / "okm_augumented_2021_preprocessed.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0])

TRAIN_END = pd.Timestamp("2021-07-01")   # 1~6월
TUNE_END = pd.Timestamp("2021-08-01")    # 7월 끝
CALIB_END = pd.Timestamp("2021-08-15")   # 8/1~8/14 threshold calibration
# CALIB_END 이후 데이터는 장기 최종 평가

RANDOM_STATE = 42
POWER_COLUMNS = ["15분", "30분", "45분", "60분"]
TARGETS = ["시간평균전력", "시간최대전력"]

# 기존 모델과 동일한 후보군
LGBM_CANDIDATES = [
    {"num_leaves": 7, "max_depth": 3, "min_child_samples": 50,
     "learning_rate": 0.05, "reg_alpha": 0.0, "reg_lambda": 0.1},
    {"num_leaves": 15, "max_depth": 5, "min_child_samples": 40,
     "learning_rate": 0.03, "reg_alpha": 0.1, "reg_lambda": 0.1},
    {"num_leaves": 31, "max_depth": 6, "min_child_samples": 30,
     "learning_rate": 0.03, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"num_leaves": 15, "max_depth": -1, "min_child_samples": 70,
     "learning_rate": 0.03, "reg_alpha": 1.0, "reg_lambda": 1.0},
]


# ============================================================
# 1. 평가 지표
# ============================================================

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def wape(y_true, y_pred):
    y_true = np.asarray(y_true)
    den = np.abs(y_true).sum()
    return float(np.abs(y_true - np.asarray(y_pred)).sum() / den * 100) if den else np.nan


def pinball_loss(y_true, y_pred, quantile):
    error = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def classification_scores(actual, predicted):
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)

    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(actual) if len(actual) else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


# ============================================================
# 2. 데이터 + Feature engineering
# ============================================================

def load_and_make_features(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "날짜", "시간", "평균", "생산량", "기온", "풍속", "습도", "강수량",
        "전기요금(계절)", "인건비", *POWER_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")

    date_text = df["날짜"].astype("Int64").astype(str).str.zfill(8)
    df["날짜_dt"] = pd.to_datetime(date_text, format="%Y%m%d", errors="raise")
    df["일시"] = df["날짜_dt"] + pd.to_timedelta(df["시간"], unit="h")
    df = df.sort_values("일시").reset_index(drop=True)

    if df["일시"].duplicated().any():
        raise ValueError("중복된 날짜·시간이 있습니다.")

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    df["시간평균전력"] = df["평균"]
    df["시간최대전력"] = df[POWER_COLUMNS].max(axis=1, skipna=False)
    df["요일"] = df["날짜_dt"].dt.weekday
    df["월"] = df["날짜_dt"].dt.month

    # --------------------------------------------------------
    # 시간 Feature
    # --------------------------------------------------------
    df["시간_sin"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["시간_cos"] = np.cos(2 * np.pi * df["시간"] / 24)
    df["요일_sin"] = np.sin(2 * np.pi * df["요일"] / 7)
    df["요일_cos"] = np.cos(2 * np.pi * df["요일"] / 7)
    df["월_sin"] = np.sin(2 * np.pi * (df["월"] - 1) / 12)
    df["월_cos"] = np.cos(2 * np.pi * (df["월"] - 1) / 12)
    df["주말"] = (df["요일"] >= 5).astype(int)
    df["근무시간"] = df["시간"].between(9, 17).astype(int)
    df["야간"] = ((df["시간"] < 8) | (df["시간"] >= 18)).astype(int)

    # --------------------------------------------------------
    # 기존 생산 Feature
    # --------------------------------------------------------
    previous_production = df["생산량"].shift(1)

    df["생산량_log1p"] = np.log1p(df["생산량"].clip(lower=0))
    df["생산량_차이_1h"] = df["생산량"] - previous_production
    df["생산량_차이_24h"] = df["생산량"] - df["생산량"].shift(24)
    df["생산량_3h평균"] = df["생산량"].rolling(3, min_periods=1).mean()
    df["생산시작"] = ((df["생산량"] > 0) & (previous_production.fillna(0) == 0)).astype(int)
    df["생산종료"] = ((df["생산량"] == 0) & (previous_production > 0)).astype(int)
    df["무생산"] = (df["생산량"] == 0).astype(int)

    # --------------------------------------------------------
    # [최종 채택 1] P50 전용: 생산전환상태
    # 0=무생산, 1=시작, 2=증가, 3=감소, 4=종료, 5=유지
    # --------------------------------------------------------
    production_diff = df["생산량_차이_1h"]
    df["생산전환상태"] = np.select(
        [
            (df["생산량"] == 0) & (previous_production.fillna(0) == 0),
            (df["생산량"] > 0) & (previous_production.fillna(0) == 0),
            (df["생산량"] > 0) & (previous_production > 0) & (production_diff > 0),
            (df["생산량"] > 0) & (previous_production > 0) & (production_diff < 0),
            (df["생산량"] == 0) & (previous_production > 0),
            (df["생산량"] > 0) & (previous_production > 0) & (production_diff == 0),
        ],
        [0, 1, 2, 3, 4, 5],
        default=0,
    ).astype(int)

    # --------------------------------------------------------
    # [최종 채택 2] 최대전력 P90 전용: 오전 x 생산변화
    # 오류가 집중됐던 09~12시 생산 변화의 크기/방향을 직접 표현
    # --------------------------------------------------------
    morning = df["시간"].between(9, 12).astype(int)
    df["오전x생산변화"] = morning * df["생산량_차이_1h"].fillna(0)

    # --------------------------------------------------------
    # 날씨 Feature
    # --------------------------------------------------------
    df["강수여부"] = (df["강수량"] > 0).astype(int)
    df["냉방도"] = (df["기온"] - 24).clip(lower=0)
    df["난방도"] = (18 - df["기온"]).clip(lower=0)
    df["기온x습도"] = df["기온"] * df["습도"]

    # --------------------------------------------------------
    # 과거 전력 Feature
    # 다음 날 예측이므로 24시간 이전 lag만 사용
    # --------------------------------------------------------
    lag_features = []

    for target, prefix in [
        ("시간평균전력", "평균전력"),
        ("시간최대전력", "최대전력"),
    ]:
        for lag in [24, 48, 72, 168]:
            name = f"{prefix}_lag{lag}"
            df[name] = df[target].shift(lag)
            df[f"{name}_결측"] = df[name].isna().astype(int)
            lag_features.extend([name, f"{name}_결측"])

        group = df.groupby("시간")[target]
        mean_name = f"{prefix}_같은시간_7일평균"
        std_name = f"{prefix}_같은시간_7일표준편차"
        max_name = f"{prefix}_같은시간_7일최대"

        df[mean_name] = group.transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
        df[std_name] = group.transform(lambda x: x.shift(1).rolling(7, min_periods=3).std())
        df[max_name] = group.transform(lambda x: x.shift(1).rolling(7, min_periods=3).max())
        lag_features.extend([mean_name, std_name, max_name])

    # --------------------------------------------------------
    # BASELINE 56개 변수
    # --------------------------------------------------------
    baseline_features = [
        "시간", "요일", "월",
        "시간_sin", "시간_cos", "요일_sin", "요일_cos", "월_sin", "월_cos",
        "주말", "근무시간", "야간",

        "생산량", "생산량_log1p", "생산량_차이_1h", "생산량_차이_24h",
        "생산량_3h평균", "생산시작", "생산종료", "무생산",

        "기온", "풍속", "습도", "강수량", "강수여부", "냉방도", "난방도", "기온x습도",
        "전기요금(계절)", "인건비",
    ] + lag_features

    source_flags = [
        col for col in ["풍속_결측", "강수량_결측", "공장인원_결측", "전력계측공백"]
        if col in df.columns
    ]
    baseline_features += source_flags

    # 목적별 Feature Set
    p50_features = baseline_features + ["생산전환상태"]

    # 평균전력 P90은 오전 interaction으로 개선되지 않아 baseline 유지
    avg_p90_features = baseline_features.copy()

    # 최대전력 P90만 interaction 채택
    max_p90_features = baseline_features + ["오전x생산변화"]

    # Peak classifier는 신규 후보가 FN을 늘려 baseline 유지
    peak_features = baseline_features.copy()

    return {
        "df": df,
        "baseline": baseline_features,
        "p50": p50_features,
        "avg_p90": avg_p90_features,
        "max_p90": max_p90_features,
        "peak": peak_features,
    }


# ============================================================
# 3. LightGBM Quantile 모델 선택 / 재학습
# ============================================================

def select_best_quantile_model(X_train, y_train, X_tune, y_tune, quantile):
    best_model = None
    best_params = None
    best_loss = np.inf

    for params in LGBM_CANDIDATES:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=2000,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            verbosity=-1,
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_tune, y_tune)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )

        pred = model.predict(X_tune, num_iteration=model.best_iteration_)
        loss = pinball_loss(y_tune, pred, quantile)

        if loss < best_loss:
            best_model = model
            best_params = params.copy()
            best_loss = loss

    return best_params, int(best_model.best_iteration_), best_loss


def refit_quantile_model(X_train, y_train, quantile, params, best_iteration):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=max(1, int(best_iteration)),
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbosity=-1,
        **params,
    )
    return model.fit(X_train, y_train)


# ============================================================
# 4. Peak classifier
# ============================================================

def select_peak_iteration(X_train, y_train, X_tune, y_tune):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_tune, y_tune)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    return int(model.best_iteration_)


def refit_peak_classifier(X_train, y_train, best_iteration):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=max(1, int(best_iteration)),
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    return model.fit(X_train, y_train)


def choose_f1_threshold(y_true, probability):
    rows = []
    for threshold in np.linspace(0.01, 0.99, 197):
        scores = classification_scores(y_true, probability >= threshold)
        rows.append((threshold, scores))

    best_threshold, best_scores = max(
        rows,
        key=lambda x: (x[1]["F1"], x[1]["Precision"], x[1]["Recall"]),
    )
    return float(best_threshold), best_scores


# ============================================================
# 5. MAIN
# ============================================================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

    bundle = load_and_make_features(DATA_PATH)
    df = bundle["df"]

    baseline_features = bundle["baseline"]
    p50_features = bundle["p50"]
    avg_p90_features = bundle["avg_p90"]
    max_p90_features = bundle["max_p90"]
    peak_features = bundle["peak"]

    train_mask = df["날짜_dt"] < TRAIN_END
    tune_mask = (df["날짜_dt"] >= TRAIN_END) & (df["날짜_dt"] < TUNE_END)
    final_train_mask = df["날짜_dt"] < TUNE_END
    calib_mask = (df["날짜_dt"] >= TUNE_END) & (df["날짜_dt"] < CALIB_END)
    test_mask = df["날짜_dt"] >= CALIB_END

    print("\n" + "=" * 92)
    print("1. 최종 모델 Feature 전략")
    print("=" * 92)
    print(f"Baseline 변수: {len(baseline_features)}개")
    print(f"P50        : {len(p50_features)}개 = baseline + 생산전환상태")
    print(f"평균 P90   : {len(avg_p90_features)}개 = baseline 유지")
    print(f"최대 P90   : {len(max_p90_features)}개 = baseline + 오전x생산변화")
    print(f"Peak       : {len(peak_features)}개 = baseline 유지")
    print("※ 생산량_차이_168h는 Peak FP를 줄였지만 FN 증가 때문에 최종 미채택")

    print("\n[시간 분할]")
    print("1~6월 학습 → 7월 선택 → 1~7월 재학습 → 8/1~8/14 threshold 보정 → 8/15~9/14 평가")

    result_rows = []
    prediction_store = {}

    # --------------------------------------------------------
    # P50 / P90
    # --------------------------------------------------------
    for target in TARGETS:
        target_ok = df[target].notna()
        tr = train_mask & target_ok
        tu = tune_mask & target_ok
        ft = final_train_mask & target_ok
        te = test_mask & target_ok

        y_train = df.loc[tr, target]
        y_tune = df.loc[tu, target]
        y_final = df.loc[ft, target]
        y_test = df.loc[te, target]

        # ---------------- P50 ----------------
        p50_params, p50_iter, p50_tune_loss = select_best_quantile_model(
            df.loc[tr, p50_features], y_train,
            df.loc[tu, p50_features], y_tune,
            0.50,
        )
        p50_model = refit_quantile_model(
            df.loc[ft, p50_features], y_final,
            0.50, p50_params, p50_iter,
        )
        p50_pred = p50_model.predict(df.loc[te, p50_features])

        # ---------------- P90 ----------------
        p90_features = max_p90_features if target == "시간최대전력" else avg_p90_features
        p90_params, p90_iter, p90_tune_loss = select_best_quantile_model(
            df.loc[tr, p90_features], y_train,
            df.loc[tu, p90_features], y_tune,
            0.90,
        )
        p90_model = refit_quantile_model(
            df.loc[ft, p90_features], y_final,
            0.90, p90_params, p90_iter,
        )
        raw_p90 = p90_model.predict(df.loc[te, p90_features])
        raw_p90 = np.maximum(raw_p90, p50_pred)

        row = {
            "대상": target,
            "P50_feature수": len(p50_features),
            "P90_feature수": len(p90_features),
            "P50_MAE": mae(y_test, p50_pred),
            "P50_RMSE": rmse(y_test, p50_pred),
            "P50_WAPE(%)": wape(y_test, p50_pred),
            "P90_Pinball": pinball_loss(y_test, raw_p90, 0.90),
            "P90_Coverage(%)": float(np.mean(np.asarray(y_test) <= raw_p90) * 100),
            "P50_tune_pinball": p50_tune_loss,
            "P90_tune_pinball": p90_tune_loss,
        }
        result_rows.append(row)

        prediction_store[target] = {
            "index": df.loc[te].index,
            "actual": y_test.to_numpy(),
            "p50": p50_pred,
            "p90": raw_p90,
        }

    results = pd.DataFrame(result_rows)

    print("\n" + "=" * 92)
    print("2. P50 / P90 최종 성능")
    print("=" * 92)
    print(results.round(4).to_string(index=False))

    # --------------------------------------------------------
    # Peak classifier
    # --------------------------------------------------------
    max_valid = df["시간최대전력"].notna()
    tr = train_mask & max_valid
    tu = tune_mask & max_valid
    ft = final_train_mask & max_valid
    ca = calib_mask & max_valid
    te = test_mask & max_valid

    # 최초 학습기간 최대전력 상위 10%를 피크로 정의
    peak_threshold = df.loc[tr, "시간최대전력"].quantile(0.90)

    y_train_peak = (df.loc[tr, "시간최대전력"] >= peak_threshold).astype(int)
    y_tune_peak = (df.loc[tu, "시간최대전력"] >= peak_threshold).astype(int)
    y_final_peak = (df.loc[ft, "시간최대전력"] >= peak_threshold).astype(int)
    y_calib_peak = (df.loc[ca, "시간최대전력"] >= peak_threshold).astype(int)
    y_test_peak = (df.loc[te, "시간최대전력"] >= peak_threshold).astype(int)

    peak_iter = select_peak_iteration(
        df.loc[tr, peak_features], y_train_peak,
        df.loc[tu, peak_features], y_tune_peak,
    )
    peak_model = refit_peak_classifier(
        df.loc[ft, peak_features], y_final_peak, peak_iter,
    )

    calib_probability = peak_model.predict_proba(df.loc[ca, peak_features])[:, 1]
    alert_threshold, calib_scores = choose_f1_threshold(y_calib_peak, calib_probability)

    test_probability = peak_model.predict_proba(df.loc[te, peak_features])[:, 1]
    test_prediction = test_probability >= alert_threshold
    peak_scores = classification_scores(y_test_peak, test_prediction)

    pr_auc = average_precision_score(y_test_peak, test_probability)
    brier = brier_score_loss(y_test_peak, test_probability)

    print("\n" + "=" * 92)
    print("3. Peak classifier 최종 성능")
    print("=" * 92)
    print(f"피크 기준: 최대전력 >= {peak_threshold:.3f}")
    print(f"8/1~8/14 선택 threshold: {alert_threshold:.3f}")
    print(f"Calibration F1={calib_scores['F1']:.3f}, Recall={calib_scores['Recall']:.3f}")
    print(
        f"Test Accuracy={peak_scores['Accuracy']:.4f} | "
        f"Precision={peak_scores['Precision']:.4f} | "
        f"Recall={peak_scores['Recall']:.4f} | F1={peak_scores['F1']:.4f}"
    )
    print(
        f"TP={peak_scores['TP']} / FP={peak_scores['FP']} / "
        f"FN={peak_scores['FN']} / TN={peak_scores['TN']}"
    )
    print(f"PR-AUC={pr_auc:.4f} | Brier={brier:.4f}")

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------
    results.to_csv(BASE_DIR / "final_regression_metrics.csv", index=False, encoding="utf-8-sig")

    peak_metrics = pd.DataFrame([{
        "peak_threshold": peak_threshold,
        "alert_threshold": alert_threshold,
        **peak_scores,
        "PR_AUC": pr_auc,
        "Brier": brier,
    }])
    peak_metrics.to_csv(BASE_DIR / "final_peak_metrics.csv", index=False, encoding="utf-8-sig")

    # 최대전력 예측값 저장: 최적화 친구에게 전달하기 쉬운 형태
    max_pred = prediction_store["시간최대전력"]
    pred_df = df.loc[max_pred["index"], ["일시", "시간", "생산량"]].copy()
    pred_df["실제_최대전력"] = max_pred["actual"]
    pred_df["P50_최대전력"] = max_pred["p50"]
    pred_df["P90_최대전력"] = max_pred["p90"]
    pred_df["피크확률"] = test_probability
    pred_df["실제_피크"] = y_test_peak.to_numpy()
    pred_df["예측_피크"] = test_prediction.astype(int)
    pred_df.to_csv(BASE_DIR / "final_maxpower_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n저장 완료:")
    print("- final_regression_metrics.csv")
    print("- final_peak_metrics.csv")
    print("- final_maxpower_predictions.csv")


if __name__ == "__main__":
    main()
