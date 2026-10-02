"""
제조 전력 예측 + 피크 경보 모델

핵심 목적
1) 다음 날 평균전력 / 최대전력의 일반적인 수준(P50) 예측
2) 높은 전력 사용 가능성을 보는 상한(P90) 예측
3) 실제 피크 발생 여부를 확률로 예측하여 경보
4) 단순 성능 숫자뿐 아니라
   - 실제 피크를 놓친 FN(False Negative)
   - 안전한데 피크라고 한 FP(False Positive)
   - 모델이 잘 작동하는 조건 / 실패하는 조건
   - 큰 오차가 반복되는 구간
   을 함께 분석

시간 분할
- 1~6월 : 모델/파라미터 탐색용 학습
- 7월   : 파라미터 + best iteration 선택
- 1~7월 : 선택된 설정으로 최종 재학습
- 8월   : P90 보정 + 피크 경보 임계값 결정
- 9월   : 최종 테스트 (절대 모델 선택에 사용하지 않음)

"""

from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 기존 폴더 구조를 우선 사용하고, 같은 폴더에 CSV가 있으면 그것도 허용
DATA_CANDIDATES = [
    BASE_DIR.parent / "data" / "okm_augumented_2021_preprocessed.csv",
    BASE_DIR / "okm_augumented_2021_preprocessed.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0])

# 날짜 분할
TRAIN_END = pd.Timestamp("2021-07-01")   # 1~6월
TUNE_END = pd.Timestamp("2021-08-01")    # 7월 끝
CALIB_END = pd.Timestamp("2021-09-01")   # 8월 끝 → 9월 테스트

RANDOM_STATE = 42
PREDICTION_ROWS_TO_SHOW = 24

# 안전 경보는 "실제 피크를 놓치는 FN"을 줄이는 것이 목적
# 0.90보다 조금 더 엄격하게 0.95로 설정.
SAFETY_MIN_RECALL = 0.95

POWER_COLUMNS = ["15분", "30분", "45분", "60분"]
TARGETS = ["시간평균전력", "시간최대전력"]

# LightGBM 후보
# 1~6월 학습 → 7월에서 비교하여 최적 조합 선택
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
# 1. 데이터 로드 + 파생변수
# ============================================================

def load_and_make_features(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "날짜", "시간", "평균", "생산량",
        "기온", "풍속", "습도", "강수량",
        "전기요금(계절)", "인건비",
        *POWER_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")

    # 날짜/시간 생성
    date_text = df["날짜"].astype("Int64").astype(str).str.zfill(8)
    df["날짜_dt"] = pd.to_datetime(date_text, format="%Y%m%d", errors="raise")
    df["일시"] = df["날짜_dt"] + pd.to_timedelta(df["시간"], unit="h")
    df = df.sort_values("일시").reset_index(drop=True)

    if df["일시"].duplicated().any():
        raise ValueError("중복된 날짜·시간이 있습니다.")

    # --------------------------------------------------------
    # 예측 대상
    # --------------------------------------------------------
    # 현재 시간의 전력값은 정답을 만드는 데만 사용한다.
    # feature로는 넣지 않기 때문에 미래정보 누수를 막는다.
    df["시간평균전력"] = df["평균"]
    df["시간최대전력"] = df[POWER_COLUMNS].max(axis=1, skipna=False)

    df["요일"] = df["날짜_dt"].dt.weekday
    df["월"] = df["날짜_dt"].dt.month

    # --------------------------------------------------------
    # 시간 파생변수
    # --------------------------------------------------------
    # 23시와 0시, 일요일과 월요일처럼 주기적으로 이어지는 특성을 표현
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
    # 생산 관련 파생변수
    # --------------------------------------------------------
    # "다음 날 생산계획을 미리 알고 있다"는 운영 가정
    df["생산량_log1p"] = np.log1p(df["생산량"].clip(lower=0))
    df["생산량_차이_1h"] = df["생산량"].diff(1)
    df["생산량_차이_24h"] = df["생산량"] - df["생산량"].shift(24)
    df["생산량_3h평균"] = df["생산량"].rolling(3, min_periods=1).mean()

    previous_production = df["생산량"].shift(1)
    df["생산시작"] = (
        (df["생산량"] > 0) &
        (previous_production.fillna(0) == 0)
    ).astype(int)

    df["생산종료"] = (
        (df["생산량"] == 0) &
        (previous_production > 0)
    ).astype(int)

    df["무생산"] = (df["생산량"] == 0).astype(int)

    # --------------------------------------------------------
    # 날씨 파생변수
    # --------------------------------------------------------
    df["강수여부"] = (df["강수량"] > 0).astype(int)
    df["냉방도"] = (df["기온"] - 24).clip(lower=0)
    df["난방도"] = (18 - df["기온"]).clip(lower=0)
    df["기온x습도"] = df["기온"] * df["습도"]

    # --------------------------------------------------------
    # 과거 전력 파생변수
    # --------------------------------------------------------
    # 다음 날 예측 관점에서 확실히 알고 있는 24시간 이전 값만 사용한다.
    # lag1, lag2, lag3 같은 당일 직전 정보는 사용하지 않는다.
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

        # 같은 시간대 최근 7일 패턴
        group = df.groupby("시간")[target]

        mean_name = f"{prefix}_같은시간_7일평균"
        std_name = f"{prefix}_같은시간_7일표준편차"
        max_name = f"{prefix}_같은시간_7일최대"

        df[mean_name] = group.transform(
            lambda x: x.shift(1).rolling(7, min_periods=3).mean()
        )
        df[std_name] = group.transform(
            lambda x: x.shift(1).rolling(7, min_periods=3).std()
        )
        df[max_name] = group.transform(
            lambda x: x.shift(1).rolling(7, min_periods=3).max()
        )

        lag_features.extend([mean_name, std_name, max_name])

    base_features = [
        # 시간
        "시간", "요일", "월",
        "시간_sin", "시간_cos",
        "요일_sin", "요일_cos",
        "월_sin", "월_cos",
        "주말", "근무시간", "야간",

        # 생산
        "생산량", "생산량_log1p",
        "생산량_차이_1h", "생산량_차이_24h",
        "생산량_3h평균",
        "생산시작", "생산종료", "무생산",

        # 날씨
        "기온", "풍속", "습도", "강수량",
        "강수여부", "냉방도", "난방도", "기온x습도",

        # 비용
        "전기요금(계절)", "인건비",
    ]

    # 전처리 파일에 존재하는 경우에만 사용
    source_flags = [
        col for col in [
            "풍속_결측", "강수량_결측",
            "공장인원_결측", "전력계측공백"
        ]
        if col in df.columns
    ]

    return df, base_features + lag_features + source_flags


# ============================================================
# 2. 평가 지표
# ============================================================

def mae(y_true, y_pred):
    """평균 절대오차: 실제 전력 단위 그대로 해석 가능."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    """큰 오차에 더 큰 패널티를 주는 지표."""
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(y_true) - np.asarray(y_pred)) ** 2
            )
        )
    )


def wape(y_true, y_pred):
    """
    전체 실제 전력 규모 대비 절대오차 비율.
    예: WAPE 6% → 전체 전력 규모 대비 약 6% 수준의 오차.
    """
    y_true = np.asarray(y_true)
    denominator = np.abs(y_true).sum()

    if denominator == 0:
        return np.nan

    return float(
        np.abs(y_true - np.asarray(y_pred)).sum()
        / denominator
        * 100
    )


def pinball_loss(y_true, y_pred, quantile):
    """P90 같은 분위수 예측용 손실. 낮을수록 좋다."""
    error = np.asarray(y_true) - np.asarray(y_pred)

    return float(
        np.mean(
            np.maximum(
                quantile * error,
                (quantile - 1) * error,
            )
        )
    )


def classification_scores(actual, predicted):
    """
    피크 분류 평가.

    가장 중요한 오류:
    FN = 실제 피크인데 모델이 안전하다고 판단한 경우.
    현장 관점에서는 이 오류가 FP(오경보)보다 더 위험하다.
    """
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)

    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / len(actual) if len(actual) else 0.0

    # 실제 피크 중 놓친 비율 = 가장 위험한 오류율
    fnr = fn / (tp + fn) if tp + fn else 0.0

    # 실제 안전 구간 중 잘못 경보한 비율
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy,
        "정밀도": precision,
        "재현율": recall,
        "F1": f1,
        "FN율": fnr,
        "FP율": fpr,
    }


# ============================================================
# 3. P90 보정 / 경보 임계값
# ============================================================

def calibrate_upper_quantile(y_calib, raw_p90):
    """
    8월 calibration 데이터에서
    실제값 - raw P90 잔차의 90% 분위수를 계산하여 P90을 보정한다.

    단, 보정이 항상 좋은 것은 아니므로
    9월 테스트에서는 raw와 calibrated를 둘 다 출력한다.
    """
    residual = np.asarray(y_calib) - np.asarray(raw_p90)

    try:
        return float(np.quantile(residual, 0.90, method="higher"))
    except TypeError:
        return float(np.quantile(residual, 0.90, interpolation="higher"))


def get_threshold_candidates(y_true, probability):
    """여러 임계값의 Precision / Recall / F1 / FN율 / FP율 계산."""
    rows = []

    for threshold in np.linspace(0.01, 0.99, 197):
        scores = classification_scores(
            y_true,
            probability >= threshold,
        )
        rows.append({
            "threshold": float(threshold),
            **scores,
        })

    return pd.DataFrame(rows)


def choose_balanced_threshold(y_true, probability):
    """
    균형 경보:
    F1 최대화.
    Precision과 Recall의 균형이 가장 좋은 임계값을 선택.
    """
    table = get_threshold_candidates(y_true, probability)

    best = table.sort_values(
        ["F1", "정밀도", "재현율"],
        ascending=False,
    ).iloc[0]

    return float(best["threshold"]), best.to_dict()


def choose_safety_threshold(
    y_true,
    probability,
    minimum_recall=SAFETY_MIN_RECALL,
):
    """
    안전 경보:
    실제 피크를 안전하다고 판단하는 FN을 줄이는 것이 최우선.

    1) Recall >= minimum_recall 조건을 만족
    2) 그 안에서 가능한 한 높은 threshold를 선택
       → Recall 조건을 유지하면서 FP를 줄이려는 목적
    3) 동률이면 F1 / Precision이 높은 쪽 선택
    """
    table = get_threshold_candidates(y_true, probability)

    feasible = table[
        table["재현율"] >= minimum_recall
    ].copy()

    # 조건을 만족하는 threshold가 없다면
    # recall이 가장 높은 후보를 사용
    if feasible.empty:
        best = table.sort_values(
            ["재현율", "F1", "정밀도"],
            ascending=False,
        ).iloc[0]
    else:
        best = feasible.sort_values(
            ["threshold", "F1", "정밀도"],
            ascending=False,
        ).iloc[0]

    return float(best["threshold"]), best.to_dict()


def print_threshold_table(y_true, probability, balanced_th, safety_th):
    """대표 임계값별 성능 + 실제 선택된 threshold 출력."""
    rows = []

    thresholds = [
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
        balanced_th,
        safety_th,
    ]

    labels = [
        "0.05", "0.10", "0.20", "0.30", "0.40", "0.50",
        "F1최적",
        f"안전(Recall>={SAFETY_MIN_RECALL:.2f})",
    ]

    for label, threshold in zip(labels, thresholds):
        s = classification_scores(
            y_true,
            probability >= threshold,
        )

        rows.append({
            "기준": label,
            "threshold": threshold,
            "Accuracy": s["Accuracy"],
            "Precision": s["정밀도"],
            "Recall": s["재현율"],
            "F1": s["F1"],
            "FN": s["FN"],
            "FP": s["FP"],
            "FN율": s["FN율"],
            "FP율": s["FP율"],
        })

    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))


# ============================================================
# 4. 베이스라인 / Gradient Boosting
# ============================================================

def seasonal_quantile_prediction(
    train_df,
    predict_df,
    target,
    quantile,
):
    """요일 × 시간 기준 단순 통계 베이스라인."""
    table = train_df.groupby(
        ["요일", "시간"]
    )[target].quantile(quantile)

    hour_table = train_df.groupby(
        "시간"
    )[target].quantile(quantile)

    global_value = train_df[target].quantile(quantile)

    keys = pd.MultiIndex.from_frame(
        predict_df[["요일", "시간"]]
    )

    pred = pd.Series(
        table.reindex(keys).to_numpy(),
        index=predict_df.index,
    )

    return (
        pred
        .fillna(predict_df["시간"].map(hour_table))
        .fillna(global_value)
        .to_numpy()
    )


def fit_gradient_quantile(X_train, y_train, quantile):
    """Gradient Boosting 비교 모델."""
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "model",
            GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=300,
                learning_rate=0.04,
                max_depth=3,
                min_samples_leaf=20,
                subsample=0.9,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return model.fit(X_train, y_train)


# ============================================================
# 5. LightGBM 선택 → 1~7월 최종 재학습
# ============================================================

def select_best_lgbm_quantile(
    X_train,
    y_train,
    X_tune,
    y_tune,
    quantile,
):
    """
    1~6월 학습, 7월 평가.

    여기서는 최종 예측을 하지 않고
    최적 hyperparameter와 best_iteration만 선택한다.
    """
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
            importance_type="gain",
            random_state=RANDOM_STATE,
            verbosity=-1,
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_tune, y_tune)],
            eval_metric="quantile",
            callbacks=[
                lgb.early_stopping(
                    80,
                    verbose=False,
                )
            ],
        )

        pred = model.predict(
            X_tune,
            num_iteration=model.best_iteration_,
        )

        loss = pinball_loss(
            y_tune,
            pred,
            quantile,
        )

        if loss < best_loss:
            best_model = model
            best_params = params.copy()
            best_loss = loss

    return (
        best_params,
        int(best_model.best_iteration_),
        best_loss,
    )


def refit_lgbm_quantile(
    X_final_train,
    y_final_train,
    quantile,
    params,
    best_iteration,
):
    """
    7월에서 선택된 설정으로 1~7월 전체를 다시 학습.
    이 단계가 기존 코드에 없었던 핵심 수정점.
    """
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=max(1, int(best_iteration)),
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        importance_type="gain",
        random_state=RANDOM_STATE,
        verbosity=-1,
        **params,
    )

    return model.fit(
        X_final_train,
        y_final_train,
    )


# ============================================================
# 6. 피크 분류기 선택 → 최종 재학습
# ============================================================

def select_peak_classifier(
    X_train,
    y_train,
    X_tune,
    y_tune,
):
    """
    피크 분류기의 iteration을
    1~6월 → 7월 기준으로 선택.
    """
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
        importance_type="gain",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_tune, y_tune)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                80,
                verbose=False,
            )
        ],
    )

    return int(model.best_iteration_)


def refit_peak_classifier(
    X_final_train,
    y_final_train,
    best_iteration,
):
    """선택된 iteration으로 1~7월 전체 재학습."""
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
        importance_type="gain",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    return model.fit(
        X_final_train,
        y_final_train,
    )


# ============================================================
# 7. 모델이 잘 되는 조건 / 실패하는 조건 분석
# ============================================================

def add_condition_columns(frame):
    """조건별 오차 분석을 위한 사람이 읽기 쉬운 구간 변수."""
    out = frame.copy()

    out["시간대"] = pd.cut(
        out["시간"],
        bins=[-1, 5, 8, 12, 17, 21, 23],
        labels=[
            "00~05",
            "06~08",
            "09~12",
            "13~17",
            "18~21",
            "22~23",
        ],
    )

    # 테스트 데이터 안에서 생산량을 저/중/고로 나눔
    positive = out.loc[out["생산량"] > 0, "생산량"]

    if positive.nunique() >= 3:
        q1, q2 = positive.quantile([0.33, 0.67])

        out["생산량구간"] = pd.cut(
            out["생산량"],
            bins=[-np.inf, 0, q1, q2, np.inf],
            labels=[
                "무생산",
                "저생산",
                "중생산",
                "고생산",
            ],
            include_lowest=True,
        )
    else:
        out["생산량구간"] = np.where(
            out["생산량"] > 0,
            "생산",
            "무생산",
        )

    return out


def analyze_regression_conditions(
    test_df,
    actual,
    prediction,
    title,
):
    """
    P50 회귀 모델이 어느 조건에서 잘 맞고,
    어느 조건에서 오차가 큰지 분석.

    단순 Top error 1~2개보다
    '반복적으로 오차가 큰 조건'을 찾는 데 초점.
    """
    work = test_df[
        [
            "일시", "시간", "요일", "주말",
            "생산량", "생산량_차이_1h",
            "생산시작", "생산종료",
            "기온", "습도",
            "최대전력_lag24", "최대전력_lag168",
        ]
    ].copy()

    work["실제값"] = np.asarray(actual)
    work["예측값"] = np.asarray(prediction)
    work["오차"] = work["실제값"] - work["예측값"]
    work["절대오차"] = np.abs(work["오차"])

    work = add_condition_columns(work)

    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

    # --------------------------------------------------------
    # 1) 시간대별
    # --------------------------------------------------------
    by_hour = (
        work.groupby("시간대", observed=True)
        .agg(
            건수=("절대오차", "size"),
            MAE=("절대오차", "mean"),
            Bias=("오차", "mean"),
            실제평균=("실제값", "mean"),
        )
        .sort_values("MAE")
    )

    print("\n[시간대별 오차 - 위쪽이 잘 작동, 아래쪽이 실패하기 쉬움]")
    print(by_hour.round(2).to_string())

    # --------------------------------------------------------
    # 2) 생산량 구간별
    # --------------------------------------------------------
    by_prod = (
        work.groupby("생산량구간", observed=True)
        .agg(
            건수=("절대오차", "size"),
            MAE=("절대오차", "mean"),
            Bias=("오차", "mean"),
            실제평균=("실제값", "mean"),
        )
        .sort_values("MAE")
    )

    print("\n[생산량 구간별 오차]")
    print(by_prod.round(2).to_string())

    # --------------------------------------------------------
    # 3) 반복적으로 실패하는 조합
    # --------------------------------------------------------
    combo = (
        work.groupby(
            ["시간대", "생산량구간"],
            observed=True,
        )
        .agg(
            건수=("절대오차", "size"),
            MAE=("절대오차", "mean"),
            RMSE=(
                "오차",
                lambda x: float(
                    np.sqrt(np.mean(np.asarray(x) ** 2))
                ),
            ),
            Bias=("오차", "mean"),
        )
        .reset_index()
    )

    # 표본이 너무 적은 조건은 우연일 수 있으므로 5건 이상만 봄
    repeated = combo[
        combo["건수"] >= 5
    ].sort_values(
        "MAE",
        ascending=False,
    )

    print("\n[반복적으로 오차가 큰 조건 Top 10 - 건수 5 이상]")
    if len(repeated):
        print(
            repeated.head(10)
            .round(2)
            .to_string(index=False)
        )
    else:
        print("표본 5건 이상인 반복 조건이 없습니다.")

    # --------------------------------------------------------
    # 4) 가장 큰 개별 오차
    # --------------------------------------------------------
    print("\n[절대오차가 가장 큰 사례 Top 10]")
    columns = [
        "일시", "시간", "생산량",
        "생산량_차이_1h",
        "기온", "습도",
        "실제값", "예측값",
        "오차", "절대오차",
    ]
    print(
        work.sort_values(
            "절대오차",
            ascending=False,
        )[columns]
        .head(10)
        .round(2)
        .to_string(index=False)
    )

    return work


def analyze_peak_errors(
    test_df,
    actual,
    probability,
    threshold,
    policy_name,
):
    """
    피크 분류의 TP / FP / FN / TN 조건 분석.

    특히 FN:
    '실제 피크인데 안전하다고 판단'
    → 현장 관점에서 가장 위험한 오류이므로 별도 출력.
    """
    prediction = probability >= threshold
    scores = classification_scores(
        actual,
        prediction,
    )

    error_df = test_df[
        [
            "일시", "시간", "요일", "주말",
            "생산량", "생산량_차이_1h",
            "생산시작", "생산종료",
            "기온", "습도",
            "최대전력_lag24",
            "최대전력_lag168",
            "시간최대전력",
        ]
    ].copy()

    error_df["실제_피크"] = np.asarray(actual).astype(int)
    error_df["예측_피크"] = prediction.astype(int)
    error_df["피크확률"] = probability

    conditions = [
        (error_df["실제_피크"] == 1) &
        (error_df["예측_피크"] == 1),

        (error_df["실제_피크"] == 0) &
        (error_df["예측_피크"] == 1),

        (error_df["실제_피크"] == 1) &
        (error_df["예측_피크"] == 0),
    ]

    error_df["분류결과"] = np.select(
        conditions,
        ["TP", "FP", "FN"],
        default="TN",
    )

    print("\n" + "=" * 88)
    print(f"피크 오류 분석 - {policy_name}")
    print("=" * 88)

    print(
        f"threshold={threshold:.3f} | "
        f"Accuracy={scores['Accuracy']:.3f} | "
        f"Precision={scores['정밀도']:.3f} | "
        f"Recall={scores['재현율']:.3f} | "
        f"F1={scores['F1']:.3f}"
    )

    print(
        f"혼동행렬: "
        f"TP={scores['TP']} / "
        f"FP={scores['FP']} / "
        f"FN={scores['FN']} / "
        f"TN={scores['TN']}"
    )

    print(
        f"FN율(실제 피크를 안전으로 놓친 비율)={scores['FN율']:.3%} | "
        f"FP율(실제 안전인데 경보한 비율)={scores['FP율']:.3%}"
    )

    # --------------------------------------------------------
    # TP / FP / FN / TN 평균 조건 비교
    # --------------------------------------------------------
    summary = (
        error_df.groupby("분류결과")
        .agg(
            건수=("분류결과", "size"),
            시간_평균=("시간", "mean"),
            생산량_평균=("생산량", "mean"),
            생산량변화_평균=("생산량_차이_1h", "mean"),
            기온_평균=("기온", "mean"),
            습도_평균=("습도", "mean"),
            lag24_평균=("최대전력_lag24", "mean"),
            lag168_평균=("최대전력_lag168", "mean"),
            실제최대_평균=("시간최대전력", "mean"),
            피크확률_평균=("피크확률", "mean"),
        )
    )

    print("\n[TP / FP / FN / TN 평균 조건]")
    print(summary.round(2).to_string())

    # --------------------------------------------------------
    # 가장 위험한 FN을 별도 출력
    # --------------------------------------------------------
    fn_df = error_df[
        error_df["분류결과"] == "FN"
    ].copy()

    print("\n[가장 중요한 오류: FN = 실제 피크인데 안전 판정]")
    if len(fn_df) == 0:
        print("FN 없음 → 테스트에서 실제 피크를 놓치지 않았습니다.")
    else:
        print(
            fn_df.sort_values(
                "시간최대전력",
                ascending=False,
            )
            .head(15)
            .round(3)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 오경보 FP도 별도 출력
    # --------------------------------------------------------
    fp_df = error_df[
        error_df["분류결과"] == "FP"
    ].copy()

    print("\n[오경보: FP = 실제 안전인데 피크 경보]")
    if len(fp_df) == 0:
        print("FP 없음")
    else:
        print(
            fp_df.sort_values(
                "피크확률",
                ascending=False,
            )
            .head(15)
            .round(3)
            .to_string(index=False)
        )

    return error_df, scores


# ============================================================
# 8. MAIN
# ============================================================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}"
        )

    df, features = load_and_make_features(DATA_PATH)

    # 시간순 분할
    train_mask = (
        (df["날짜_dt"] < TRAIN_END)
    )
    tune_mask = (
        (df["날짜_dt"] >= TRAIN_END) &
        (df["날짜_dt"] < TUNE_END)
    )
    final_train_mask = (
        df["날짜_dt"] < TUNE_END
    )
    calib_mask = (
        (df["날짜_dt"] >= TUNE_END) &
        (df["날짜_dt"] < CALIB_END)
    )
    test_mask = (
        df["날짜_dt"] >= CALIB_END
    )

    print("\n" + "=" * 88)
    print("1. 데이터 / 모델링 구조")
    print("=" * 88)

    print(f"데이터: {DATA_PATH}")
    print(f"입력변수: {len(features)}개")
    print(
        "\n1~6월: 모델 선택용 학습\n"
        "7월  : hyperparameter + iteration 선택\n"
        "1~7월: 최종 재학습\n"
        "8월  : P90 보정 + 피크 threshold 결정\n"
        "9월  : 최종 테스트"
    )

    prediction_store = {}
    all_results = []
    final_p50_models = {}

    # ========================================================
    # 9. 평균전력 / 최대전력 P50 + P90
    # ========================================================
    for target in TARGETS:
        valid_target = df[target].notna()

        tr = train_mask & valid_target
        tu = tune_mask & valid_target
        ft = final_train_mask & valid_target
        ca = calib_mask & valid_target
        te = test_mask & valid_target

        train_df = df.loc[tr]
        tune_df = df.loc[tu]
        final_train_df = df.loc[ft]
        calib_df = df.loc[ca]
        test_df = df.loc[te]

        X_train = train_df[features]
        y_train = train_df[target]

        X_tune = tune_df[features]
        y_tune = tune_df[target]

        X_final_train = final_train_df[features]
        y_final_train = final_train_df[target]

        X_calib = calib_df[features]
        y_calib = calib_df[target]

        X_test = test_df[features]
        y_test = test_df[target]

        # ----------------------------------------------------
        # A. 통계 베이스라인
        # 공정 비교의 공정성을 위해 1~7월 데이터 사용
        # ----------------------------------------------------
        base_p50 = seasonal_quantile_prediction(
            final_train_df,
            test_df,
            target,
            0.50,
        )

        base_raw_p90 = seasonal_quantile_prediction(
            final_train_df,
            test_df,
            target,
            0.90,
        )

        base_calib_raw = seasonal_quantile_prediction(
            final_train_df,
            calib_df,
            target,
            0.90,
        )

        base_offset = calibrate_upper_quantile(
            y_calib,
            base_calib_raw,
        )

        base_cal_p90 = np.maximum(
            base_raw_p90 + base_offset,
            base_p50,
        )

        base_raw_p90 = np.maximum(
            base_raw_p90,
            base_p50,
        )

        # ----------------------------------------------------
        # B. Gradient Boosting
        # 1~7월로 최종 학습
        # ----------------------------------------------------
        gb_p50_model = fit_gradient_quantile(
            X_final_train,
            y_final_train,
            0.50,
        )

        gb_p90_model = fit_gradient_quantile(
            X_final_train,
            y_final_train,
            0.90,
        )

        gb_p50 = gb_p50_model.predict(X_test)
        gb_raw_p90 = gb_p90_model.predict(X_test)

        gb_calib_raw = gb_p90_model.predict(X_calib)
        gb_offset = calibrate_upper_quantile(
            y_calib,
            gb_calib_raw,
        )

        gb_cal_p90 = np.maximum(
            gb_raw_p90 + gb_offset,
            gb_p50,
        )

        gb_raw_p90 = np.maximum(
            gb_raw_p90,
            gb_p50,
        )

        # ----------------------------------------------------
        # C. LightGBM
        # 1~6월 → 7월에서 설정 선택
        # ----------------------------------------------------
        (
            p50_params,
            p50_iteration,
            p50_tune_loss,
        ) = select_best_lgbm_quantile(
            X_train,
            y_train,
            X_tune,
            y_tune,
            0.50,
        )

        (
            p90_params,
            p90_iteration,
            p90_tune_loss,
        ) = select_best_lgbm_quantile(
            X_train,
            y_train,
            X_tune,
            y_tune,
            0.90,
        )

        # 1~7월 최종 재학습
        lgb_p50_model = refit_lgbm_quantile(
            X_final_train,
            y_final_train,
            0.50,
            p50_params,
            p50_iteration,
        )

        lgb_p90_model = refit_lgbm_quantile(
            X_final_train,
            y_final_train,
            0.90,
            p90_params,
            p90_iteration,
        )

        final_p50_models[target] = lgb_p50_model

        # 9월 P50 / raw P90
        lgb_p50 = lgb_p50_model.predict(X_test)
        lgb_raw_p90 = lgb_p90_model.predict(X_test)

        # 8월에서 P90 offset 계산
        lgb_calib_raw = lgb_p90_model.predict(X_calib)
        lgb_offset = calibrate_upper_quantile(
            y_calib,
            lgb_calib_raw,
        )

        # 보정 P90
        lgb_cal_p90 = np.maximum(
            lgb_raw_p90 + lgb_offset,
            lgb_p50,
        )

        lgb_raw_p90 = np.maximum(
            lgb_raw_p90,
            lgb_p50,
        )

        print("\n" + "-" * 88)
        print(f"[{target}] LightGBM 선택 결과")
        print(
            f"P50: 7월 pinball={p50_tune_loss:.4f}, "
            f"iteration={p50_iteration}, "
            f"params={p50_params}"
        )
        print(
            f"P90: 7월 pinball={p90_tune_loss:.4f}, "
            f"iteration={p90_iteration}, "
            f"params={p90_params}"
        )
        print(
            f"P90 8월 calibration offset = {lgb_offset:+.4f}"
        )

        # ----------------------------------------------------
        # P50 성능 + Raw/보정 P90 둘 다 저장
        # ----------------------------------------------------
        model_bundle = {
            "통계_베이스라인": (
                base_p50,
                base_raw_p90,
                base_cal_p90,
                base_offset,
            ),
            "GradientBoosting": (
                gb_p50,
                gb_raw_p90,
                gb_cal_p90,
                gb_offset,
            ),
            "LightGBM": (
                lgb_p50,
                lgb_raw_p90,
                lgb_cal_p90,
                lgb_offset,
            ),
        }

        for model_name, (
            pred_p50,
            raw_p90,
            calibrated_p90,
            offset,
        ) in model_bundle.items():

            all_results.append({
                "대상": target,
                "모델": model_name,

                # P50 회귀 성능
                "P50_MAE": mae(y_test, pred_p50),
                "P50_RMSE": rmse(y_test, pred_p50),
                "P50_WAPE(%)": wape(y_test, pred_p50),

                # P90 원본
                "Raw_P90_Pinball": pinball_loss(
                    y_test,
                    raw_p90,
                    0.90,
                ),
                "Raw_P90_Coverage(%)": float(
                    np.mean(
                        np.asarray(y_test) <= raw_p90
                    ) * 100
                ),

                # P90 보정
                "Cal_P90_Pinball": pinball_loss(
                    y_test,
                    calibrated_p90,
                    0.90,
                ),
                "Cal_P90_Coverage(%)": float(
                    np.mean(
                        np.asarray(y_test) <= calibrated_p90
                    ) * 100
                ),

                "P90_보정값": offset,
            })

            prediction_store[
                (target, model_name)
            ] = {
                "index": test_df.index,
                "p50": pred_p50,
                "raw_p90": raw_p90,
                "cal_p90": calibrated_p90,
            }

    # ========================================================
    # 10. 회귀 성능 요약
    # ========================================================
    results = pd.DataFrame(all_results)
    results.to_csv(BASE_DIR / "model_results.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("2. 최종 테스트 회귀 성능")
    print("=" * 88)

    print(
        results.round(4)
        .to_string(index=False)
    )

    print(
        "\n※ 회귀에는 Accuracy/F1을 쓰지 않습니다. "
        "평균·최대전력 예측은 MAE / RMSE / WAPE로 평가합니다."
    )

    # ========================================================
    # 11. 최대전력 피크 정의
    # ========================================================
    max_train_valid = train_mask & df["시간최대전력"].notna()
    max_tune_valid = tune_mask & df["시간최대전력"].notna()
    max_final_valid = final_train_mask & df["시간최대전력"].notna()
    max_calib_valid = calib_mask & df["시간최대전력"].notna()
    max_test_valid = test_mask & df["시간최대전력"].notna()

    # 피크 기준은 최초 학습기간(1~6월) 최대전력의 상위 10%
    peak_threshold = df.loc[
        max_train_valid,
        "시간최대전력",
    ].quantile(0.90)

    print("\n" + "=" * 88)
    print("3. 피크 정의")
    print("=" * 88)
    print(
        f"피크 기준 = 1~6월 최대전력의 상위 10% "
        f"= {peak_threshold:.3f}"
    )

    # ========================================================
    # 12. 피크 확률 분류기
    # ========================================================
    X_cls_train = df.loc[
        max_train_valid,
        features,
    ]

    X_cls_tune = df.loc[
        max_tune_valid,
        features,
    ]

    X_cls_final = df.loc[
        max_final_valid,
        features,
    ]

    X_cls_calib = df.loc[
        max_calib_valid,
        features,
    ]

    X_cls_test = df.loc[
        max_test_valid,
        features,
    ]

    y_cls_train = (
        df.loc[
            max_train_valid,
            "시간최대전력",
        ] >= peak_threshold
    ).astype(int)

    y_cls_tune = (
        df.loc[
            max_tune_valid,
            "시간최대전력",
        ] >= peak_threshold
    ).astype(int)

    y_cls_final = (
        df.loc[
            max_final_valid,
            "시간최대전력",
        ] >= peak_threshold
    ).astype(int)

    y_cls_calib = (
        df.loc[
            max_calib_valid,
            "시간최대전력",
        ] >= peak_threshold
    ).astype(int)

    y_cls_test = (
        df.loc[
            max_test_valid,
            "시간최대전력",
        ] >= peak_threshold
    ).astype(int)

    # 1~6월 → 7월에서 best iteration 선택
    cls_best_iteration = select_peak_classifier(
        X_cls_train,
        y_cls_train,
        X_cls_tune,
        y_cls_tune,
    )

    # 1~7월 전체로 최종 재학습
    classifier = refit_peak_classifier(
        X_cls_final,
        y_cls_final,
        cls_best_iteration,
    )

    # 8월 확률로 threshold 선택
    calib_probability = classifier.predict_proba(
        X_cls_calib
    )[:, 1]

    balanced_th, balanced_calib_score = choose_balanced_threshold(
        y_cls_calib,
        calib_probability,
    )

    safety_th, safety_calib_score = choose_safety_threshold(
        y_cls_calib,
        calib_probability,
        SAFETY_MIN_RECALL,
    )

    # 9월은 오직 최종 평가
    test_probability = classifier.predict_proba(
        X_cls_test
    )[:, 1]

    balanced_pred = (
        test_probability >= balanced_th
    )

    safety_pred = (
        test_probability >= safety_th
    )

    balanced_scores = classification_scores(
        y_cls_test,
        balanced_pred,
    )

    safety_scores = classification_scores(
        y_cls_test,
        safety_pred,
    )

    print("\n" + "=" * 88)
    print("4. 피크 분류 성능")
    print("=" * 88)

    print(
        f"최종 classifier iteration = {cls_best_iteration}"
    )

    print(
        f"\n[8월에서 선택된 균형 경보 threshold] "
        f"{balanced_th:.3f}"
    )
    print(
        f"8월 F1={balanced_calib_score['F1']:.3f}, "
        f"Recall={balanced_calib_score['재현율']:.3f}"
    )

    print(
        f"\n[8월에서 선택된 안전 경보 threshold] "
        f"{safety_th:.3f}"
    )
    print(
        f"8월 F1={safety_calib_score['F1']:.3f}, "
        f"Recall={safety_calib_score['재현율']:.3f}"
    )

    # 테스트 전체 threshold 표
    print("\n[9월 threshold별 성능 비교]")
    print_threshold_table(
        y_cls_test,
        test_probability,
        balanced_th,
        safety_th,
    )

    # PR-AUC / Brier는 threshold와 무관한 확률 성능
    pr_auc = average_precision_score(
        y_cls_test,
        test_probability,
    )

    brier = brier_score_loss(
        y_cls_test,
        test_probability,
    )

    print(
        f"\n9월 PR-AUC={pr_auc:.4f} | "
        f"Brier={brier:.4f}"
    )

    # ========================================================
    # 13. 혼동행렬 + FN / FP 분석
    # ========================================================
    test_peak_df = df.loc[
        max_test_valid
    ].copy()

    balanced_error_df, balanced_scores = analyze_peak_errors(
        test_peak_df,
        y_cls_test,
        test_probability,
        balanced_th,
        "균형 경보(F1 최대)",
    )

    safety_error_df, safety_scores = analyze_peak_errors(
        test_peak_df,
        y_cls_test,
        test_probability,
        safety_th,
        f"안전 경보(Recall>={SAFETY_MIN_RECALL:.2f})",
    )

    # ========================================================
    # 14. P50 모델이 잘 되는 조건 / 실패 조건
    # ========================================================
    max_lgb = prediction_store[
        ("시간최대전력", "LightGBM")
    ]

    max_test_df = df.loc[
        max_test_valid
    ].copy()

    max_actual = max_test_df[
        "시간최대전력"
    ].to_numpy()

    regression_error_df = analyze_regression_conditions(
        max_test_df,
        max_actual,
        max_lgb["p50"],
        "5. 최대전력 P50 - 잘 작동하는 조건 / 실패하는 조건",
    )

    # ========================================================
    # 15. 성능을 올려도 남는 큰 오차에 집중
    # ========================================================
    # 테스트 절대오차 상위 10%를 '잔여 고오차'로 정의
    high_error_cut = regression_error_df[
        "절대오차"
    ].quantile(0.90)

    hard_cases = regression_error_df[
        regression_error_df["절대오차"] >= high_error_cut
    ].copy()

    print("\n" + "=" * 88)
    print("6. 성능 개선 후에도 계속 남는 고오차 구간")
    print("=" * 88)

    print(
        f"절대오차 상위 10% 기준 = "
        f"{high_error_cut:.3f}"
    )
    print(
        f"고오차 사례 = "
        f"{len(hard_cases)} / {len(regression_error_df)}건"
    )

    if len(hard_cases):
        # 어떤 조건이 고오차 사례에서 반복되는지 확인
        hard_summary = (
            hard_cases.groupby(
                ["시간대", "생산량구간"],
                observed=True,
            )
            .agg(
                건수=("절대오차", "size"),
                평균절대오차=("절대오차", "mean"),
                평균실제값=("실제값", "mean"),
                평균예측값=("예측값", "mean"),
                평균생산량=("생산량", "mean"),
                평균생산량변화=("생산량_차이_1h", "mean"),
            )
            .reset_index()
            .sort_values(
                ["건수", "평균절대오차"],
                ascending=False,
            )
        )

        print(
            "\n[고오차가 반복되는 시간대 × 생산량 조건]"
        )
        print(
            hard_summary.head(15)
            .round(2)
            .to_string(index=False)
        )

    # ========================================================
    # 16. 변수 중요도
    # ========================================================
    gain = pd.Series(
        classifier.feature_importances_,
        index=features,
    ).sort_values(
        ascending=False
    )

    if gain.sum() > 0:
        gain = gain / gain.sum() * 100

    print("\n" + "=" * 88)
    print("7. 피크 분류기 Gain 중요도 Top 15")
    print("=" * 88)
    print(
        gain.head(15)
        .round(2)
        .to_string()
    )

    # 최대전력 P50 LightGBM permutation importance
    p50_model = final_p50_models[
        "시간최대전력"
    ]

    perm = permutation_importance(
        p50_model,
        X_cls_calib,
        df.loc[
            max_calib_valid,
            "시간최대전력",
        ],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    permutation = pd.Series(
        perm.importances_mean,
        index=features,
    ).sort_values(
        ascending=False
    )

    print("\n" + "=" * 88)
    print("8. 최대전력 P50 Permutation Importance Top 15")
    print("=" * 88)
    print(
        permutation.head(15)
        .round(4)
        .to_string()
    )

    # ========================================================
    # 17. 최종 요약
    # ========================================================
    lightgbm_rows = results[
        results["모델"] == "LightGBM"
    ].copy()

    print("\n" + "=" * 88)
    print("9. 최종 핵심 성능 요약")
    print("=" * 88)

    print(
        lightgbm_rows[
            [
                "대상",
                "P50_MAE",
                "P50_WAPE(%)",
                "Raw_P90_Pinball",
                "Raw_P90_Coverage(%)",
                "Cal_P90_Pinball",
                "Cal_P90_Coverage(%)",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    peak_summary = pd.DataFrame([
        {
            "정책": "균형 경보",
            "Accuracy": balanced_scores["Accuracy"],
            "Precision": balanced_scores["정밀도"],
            "Recall": balanced_scores["재현율"],
            "F1": balanced_scores["F1"],
            "FN": balanced_scores["FN"],
            "FP": balanced_scores["FP"],
            "FN율": balanced_scores["FN율"],
            "FP율": balanced_scores["FP율"],
        },
        {
            "정책": "안전 경보",
            "Accuracy": safety_scores["Accuracy"],
            "Precision": safety_scores["정밀도"],
            "Recall": safety_scores["재현율"],
            "F1": safety_scores["F1"],
            "FN": safety_scores["FN"],
            "FP": safety_scores["FP"],
            "FN율": safety_scores["FN율"],
            "FP율": safety_scores["FP율"],
        },
    ])

    print("\n[피크 경보 최종 요약]")
    print(
        peak_summary
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # 18. 마지막 예측 샘플
    # ========================================================
    sample = max_test_df[
        ["일시", "시간최대전력"]
    ].copy()

    sample["P50"] = max_lgb["p50"]
    sample["P90_raw"] = max_lgb["raw_p90"]
    sample["P90_calibrated"] = max_lgb["cal_p90"]
    sample["피크확률"] = test_probability
    sample["실제_피크"] = y_cls_test.to_numpy()
    sample["균형경보"] = balanced_pred.astype(int)
    sample["안전경보"] = safety_pred.astype(int)
    
    sample.to_csv(BASE_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    if PREDICTION_ROWS_TO_SHOW is not None:
        sample = sample.tail(
            PREDICTION_ROWS_TO_SHOW
        )

    print("\n" + "=" * 88)
    print("10. 테스트 예측 샘플")
    print("=" * 88)

    print(
        sample.round(3)
        .to_string(
            index=False,
            na_rep="NA",
        )
    )

    print(
        "\n완료.\n"
        "- 9월 테스트 데이터는 모델/threshold 선택에 사용하지 않았습니다.\n"
        "- FN은 '실제 피크인데 안전 판정'이므로 안전 관점에서 가장 중요한 오류입니다.\n"
        "- FP는 오경보이므로 FN을 충분히 낮춘 뒤 줄이는 방향으로 봅니다.\n"
        "- 큰 오차가 반복되는 시간/생산 조건은 다음 feature 개선의 우선 대상입니다."
    )

if __name__ == "__main__":
    main()
