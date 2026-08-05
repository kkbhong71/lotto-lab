# =============================================================================
# 🎰 Lotto Analytics Dashboard - INTEGRATED EDITION v1.1
# =============================================================================
# [v1.1 추가 사항]
#
# [신규 ①] PredictionTracker — 사전 예측 성과 추적 시스템
#    - 매 실행 시 예측 세트를 prediction_log.csv 에 자동 기록
#      (기록 시각, 대상 회차, seed, 엔진 버전 → 증거의 무결성 확보)
#    - 다음 회차 파일 제공 시 지난 예측을 자동 대조하여 일치 수 기입
#    - 같은 대상 회차 중복 기록 방지 (재실행해도 기존 기록 보존)
#    - 누적 성과 보고: 평균 일치 + 부트스트랩 95% CI vs 이론 기대 0.800
#    ⚠️ Colab 은 세션 종료 시 파일이 사라지므로 prediction_log.csv 를
#       매 세션 종료 전 다운로드 → 다음 세션에 CSV와 함께 재업로드할 것
#       (또는 Google Drive 마운트 경로를 log_path 로 지정)
#
# [신규 ②] 부트스트랩 신뢰구간 판정 (LOTTO LAB Phase 2 방법론 이식)
#    - 기존 임의 ±5% 문턱 → 랜덤 귀무분포 95% 구간 기반 판정으로 교체
#    - 시스템 평균의 백분위(percentile)와 95% CI 함께 보고
#
# [개선 ③] 랜덤 대조군 확대: 라운드당 5세트 → 100세트 (rand_per_round)
#    - 대조군 추정치의 분산 축소 → 판정 안정화
#    - 라운드 최대/3+ 비교는 공정성을 위해 시스템과 동일 세트 수만 사용
# =============================================================================
# =============================================================================
# [기반] ULTIMATE EDITION v3.4 (클러스터 다양성 2단계 해결 포함)
#
# [통합 ①] v1.2 Ultimate Hybrid의 6단계 방어 데이터 로더 이식
#    - 유령 열(줄 끝 쉼표로 생긴 Unnamed) 제거
#    - 빈 행 제거 / num1~num6 이름 우선 열 선택
#    - 문자·공백 → 숫자 강제 변환 / NaN 행 리포트 후 제외
#    - 로또 규칙 검증 (1~45 범위, 중복 없는 6개)
#    - round 열 존재 시 오름차순 정렬 보장 (walk-forward 전제 조건)
#    → DataEngine / BacktestEngine / main 의 CSV 진입점 전부 일원화
#
# [통합 ②] v2.3 Brother System의 랜덤 대조군을 백테스트에 내장
#    - 매 검증 회차마다 동일 세트 수의 순수 랜덤 조합 생성
#    - 시스템 vs 랜덤: 평균 일치 / 라운드별 최대 일치 / 3+ 일치율 직접 비교
#    - "이론적 랜덤 기대값"이 아닌 "실측 랜덤"과의 공정 비교
#
# [통합 ③] 재현성 강화: np.random + random 모듈 동시 시딩
# =============================================================================

import numpy as np
import pandas as pd
import random
import os
from datetime import datetime
from collections import Counter, defaultdict
from itertools import combinations
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

warnings.filterwarnings("ignore")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ sklearn 미설치 — 간이 클러스터링 모드로 전환합니다.")


# =============================================================================
# 전역 상수
# =============================================================================
PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

class C:
    H  = '\033[95m'; B  = '\033[94m'; CN = '\033[96m'; G  = '\033[92m'
    Y  = '\033[93m'; R  = '\033[91m'; E  = '\033[0m';  BD = '\033[1m'
    DM = '\033[2m';  M  = '\033[35m'



# =============================================================================
# 0. 안전한 데이터 로더 (v1.2 이식 — 6단계 방어 + walk-forward 정렬 보장)
# =============================================================================
def load_history_ex(csv_path: str, verbose: bool = True):
    """(rounds, history) 반환 — rounds 는 round 열이 없으면 None 리스트"""
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"  📄 원본 데이터: {df.shape[0]}행 × {df.shape[1]}열")

    # [방어 ①] 전부 비어 있는 '유령 열' 제거 (줄 끝 쉼표로 생긴 Unnamed 열)
    ghost_cols = [c for c in df.columns if df[c].isna().all()]
    if ghost_cols:
        if verbose: print(f"  🧹 빈 열 {len(ghost_cols)}개 제거: {ghost_cols}")
        df = df.drop(columns=ghost_cols)

    # [방어 ②] 전부 비어 있는 행 제거
    before = len(df)
    df = df.dropna(how='all')
    if before != len(df) and verbose:
        print(f"  🧹 빈 행 {before - len(df)}개 제거")

    # [방어 ⑦·신규] round 열 존재 시 오름차순 정렬 — walk-forward의 전제 조건
    round_cols = [c for c in df.columns if str(c).strip().lower() == 'round']
    if round_cols:
        df = df.sort_values(round_cols[0], ascending=True).reset_index(drop=True)

    # [방어 ③] 번호 열 선택 — num1~num6 이름 우선, 없으면 뒤 6개
    num_cols = [c for c in df.columns if str(c).strip().lower().startswith('num')]
    if len(num_cols) == 6:
        num_df = df[num_cols]
    else:
        num_df = df.iloc[:, -6:]
        if verbose: print(f"  🎯 번호 열(위치 기준): {list(num_df.columns)}")

    # [방어 ④] 문자·공백을 숫자로 강제 변환
    num_df = num_df.apply(pd.to_numeric, errors='coerce')

    # [방어 ⑤] NaN 잔여 행 리포트 후 제외
    bad = num_df[num_df.isna().any(axis=1)]
    if len(bad) > 0 and verbose:
        print(f"  ⚠️ 결측치 포함 {len(bad)}개 행 제외 (인덱스: {list(bad.index)[:10]})")
    num_df = num_df.dropna().astype(int)

    # [방어 ⑥] 로또 규칙 검증 — 1~45 범위, 중복 없는 6개
    valid = num_df.apply(lambda r: r.between(1, 45).all() and r.nunique() == 6, axis=1)
    if int((~valid).sum()) > 0 and verbose:
        print(f"  ⚠️ 규칙 위반(범위 이탈/중복) {int((~valid).sum())}개 행 제외")
    num_df = num_df[valid]

    if len(num_df) == 0:
        raise ValueError("유효한 회차 데이터가 0건입니다. CSV 열 구조를 확인하세요.")

    if round_cols:
        r_ser  = pd.to_numeric(df.loc[num_df.index, round_cols[0]], errors='coerce')
        rounds = [int(r) if pd.notna(r) else None for r in r_ser]
    else:
        rounds = [None] * len(num_df)
    return rounds, [sorted(row) for row in num_df.values.tolist()]


def load_history(csv_path: str, verbose: bool = True) -> List[List[int]]:
    return load_history_ex(csv_path, verbose)[1]


# =============================================================================
# 0-B. 부트스트랩 통계 (LOTTO LAB Phase 2 방법론 이식)
# =============================================================================
RANDOM_EXPECTED = 6 * 6 / 45          # 세트당 이론적 기대 일치 수 = 0.800

def bootstrap_mean_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                      seed: int = 42):
    """표본 평균의 부트스트랩 (1-alpha) 신뢰구간"""
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return (0.0, 0.0)
    rng   = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))

def null_distribution_verdict(sys_matches, rand_pool,
                              n_boot: int = 2000, seed: int = 42):
    """
    귀무가설: 시스템 = 랜덤.
    랜덤 풀에서 시스템과 같은 크기의 표본 평균 분포(귀무분포)를 만들고
    시스템 평균이 그 분포의 어디에 위치하는지로 판정한다.
    반환: (null_lo, null_hi, percentile, verdict문자열)
    """
    sys_avg = float(np.mean(sys_matches))
    pool    = np.asarray(rand_pool, dtype=float)
    if len(pool) == 0:
        return (0.0, 0.0, 50.0, "판정 불가 (대조군 없음)")
    rng        = np.random.default_rng(seed)
    null_means = rng.choice(pool, size=(n_boot, len(sys_matches)),
                            replace=True).mean(axis=1)
    lo, hi = (float(np.percentile(null_means, 2.5)),
              float(np.percentile(null_means, 97.5)))
    pct    = float((null_means < sys_avg).mean() * 100)
    if sys_avg > hi:
        verdict = "시스템 우세 (95% 유의)"
    elif sys_avg < lo:
        verdict = "시스템 열세 (95% 유의)"
    else:
        verdict = "통계적 동률 — 랜덤 귀무분포 95% 구간 내"
    return lo, hi, pct, verdict


# =============================================================================
# 0-C. PredictionTracker — 사전 예측 성과 추적
# =============================================================================
class PredictionTracker:
    """
    예측을 CSV 로그에 기록하고, 새 당첨 데이터 제공 시 자동 대조한다.
    백테스트(과거 시뮬레이션)와 달리 '그 시점에 실제로 예측했다'는
    사전 등록 증거를 남기는 것이 목적이다.
    """
    COLS = ['logged_at', 'engine_version', 'seed', 'target_round', 'set_no',
            'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'match_count']
    VERSION = 'INTEGRATED v1.1'

    def __init__(self, log_path: str):
        self.log_path = log_path
        if os.path.exists(log_path):
            self.log = pd.read_csv(log_path)
            for c in self.COLS:
                if c not in self.log.columns:
                    self.log[c] = np.nan
        else:
            self.log = pd.DataFrame(columns=self.COLS)
            print(f"  📔 새 예측 로그 생성: {log_path}")
            print(f"     ⚠️ Colab 사용 시 세션 종료 전 이 파일을 꼭 다운로드하세요.")

    def _save(self):
        self.log.to_csv(self.log_path, index=False)

    # ── 기록 ──────────────────────────────────────────────────────────
    def log_predictions(self, target_round, results, seed) -> bool:
        if target_round is None:
            print(f"  📔 round 열이 없어 대상 회차를 특정할 수 없습니다. 기록 생략.")
            return False
        existing = self.log[self.log['target_round'] == target_round]
        if len(existing) > 0:
            print(f"  📔 {target_round}회차 예측은 이미 {len(existing)}세트 기록됨 "
                  f"— 무결성 보호를 위해 재기록하지 않습니다.")
            return False
        now  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rows = []
        for i, r in enumerate(results, 1):
            n = sorted(r['nums'])
            rows.append({'logged_at': now, 'engine_version': self.VERSION,
                         'seed': seed, 'target_round': target_round,
                         'set_no': i,
                         'n1': n[0], 'n2': n[1], 'n3': n[2],
                         'n4': n[3], 'n5': n[4], 'n6': n[5],
                         'match_count': np.nan})
        self.log = pd.concat([self.log, pd.DataFrame(rows)], ignore_index=True)
        self._save()
        print(f"  📔 {target_round}회차 예측 {len(rows)}세트 기록 완료 → {self.log_path}")
        return True

    # ── 대조 ──────────────────────────────────────────────────────────
    def reconcile(self, round_map: Dict[int, List[int]]) -> int:
        """미대조 예측을 실제 당첨번호와 대조. 대조된 세트 수 반환."""
        pending = self.log['match_count'].isna()
        updated = 0
        for idx in self.log[pending].index:
            tr = self.log.at[idx, 'target_round']
            tr = int(tr) if pd.notna(tr) else None
            if tr in round_map:
                actual = set(round_map[tr])
                pred   = {int(self.log.at[idx, f'n{k}']) for k in range(1, 7)}
                self.log.at[idx, 'match_count'] = len(pred & actual)
                updated += 1
        if updated:
            self._save()
            done = self.log[self.log['match_count'].notna()]
            rounds_done = sorted(int(r) for r in done['target_round'].unique())
            print(f"  🔗 지난 예측 {updated}세트 자동 대조 완료 "
                  f"(회차: {rounds_done[-3:] if len(rounds_done)>3 else rounds_done})")
        return updated

    # ── 누적 성과 보고 ────────────────────────────────────────────────
    def report(self):
        done = self.log[self.log['match_count'].notna()]
        n_pending = int(self.log['match_count'].isna().sum())
        if len(done) == 0:
            if n_pending:
                print(f"  📔 대기 중 예측 {n_pending}세트 "
                      f"(다음 회차 파일 제공 시 자동 대조)")
            return
        matches  = done['match_count'].astype(float).values
        n_rounds = done['target_round'].nunique()
        avg      = float(np.mean(matches))
        lo, hi   = bootstrap_mean_ci(matches)
        dist     = Counter(int(m) for m in matches)
        print(f"\n  {'═'*55}")
        print(f"  📈 사전 예측 누적 성과  ({n_rounds}회차 · {len(matches)}세트)")
        print(f"  {'═'*55}")
        print(f"  평균 일치      : {avg:.3f}  [95% CI: {lo:.3f} ~ {hi:.3f}]")
        print(f"  이론 랜덤 기대 : {RANDOM_EXPECTED:.3f}")
        for mc in sorted(dist.keys(), reverse=True):
            print(f"    {mc}개 일치: {dist[mc]}세트")
        if lo > RANDOM_EXPECTED:
            print(f"  ⚖️  판정: 랜덤 기대 초과 (95% 유의) — 표본 확대 후 재검증 권장")
        elif hi < RANDOM_EXPECTED:
            print(f"  ⚖️  판정: 랜덤 기대 미달 (95% 유의)")
        else:
            print(f"  ⚖️  판정: 통계적 동률 — CI가 이론 기대 {RANDOM_EXPECTED:.3f}을 포함")
        if n_rounds < 50:
            print(f"  💡 통계적 유의성 목표(50회차)까지 {50 - n_rounds}회차 남음")
        if n_pending:
            print(f"  📔 대기 중 예측 {n_pending}세트")


# =============================================================================
# 1. Config
# =============================================================================
@dataclass
class Config:
    SUM_RANGE:            Tuple[int, int] = (95, 200)
    SUM_OPTIMAL:          Tuple[int, int] = (110, 190)
    MIN_AC:               int             = 7
    ODD_RATES:            List[int]       = field(default_factory=lambda: [2, 3, 4])
    HIGH_RATES:           List[int]       = field(default_factory=lambda: [2, 3, 4])
    MIN_ZONES:            int             = 3
    MAX_PER_ZONE:         int             = 3
    MIN_UNIQUE_ENDINGS:   int             = 3
    MAX_SAME_ENDING:      int             = 2
    PRIME_RANGE:          Tuple[int, int] = (1, 4)
    MAX_CONSECUTIVE:      int             = 2
    PREV_MATCH_RANGE:     Tuple[int, int] = (0, 3)
    WEIGHT_FREQUENCY:     float           = 0.18
    WEIGHT_RECENCY:       float           = 0.18
    WEIGHT_GAP:           float           = 0.16
    WEIGHT_PAIR:          float           = 0.14
    WEIGHT_MOMENTUM:      float           = 0.20
    WEIGHT_ZONE:          float           = 0.14
    RECENT_PERIODS:       List[int]       = field(default_factory=lambda: [10, 30, 50, 100])
    N_CLUSTERS:           int             = 5
    CLUSTER_BONUS_WEIGHT: float           = 0.10
    MAX_SET_OVERLAP:      int             = 1

    def get_weights_dict(self) -> Dict[str, float]:
        return {
            'frequency': self.WEIGHT_FREQUENCY,
            'recency':   self.WEIGHT_RECENCY,
            'gap':       self.WEIGHT_GAP,
            'pair':      self.WEIGHT_PAIR,
            'momentum':  self.WEIGHT_MOMENTUM,
            'zone':      self.WEIGHT_ZONE,
        }

    def set_weights_from_dict(self, w: Dict[str, float]):
        self.WEIGHT_FREQUENCY = w.get('frequency', self.WEIGHT_FREQUENCY)
        self.WEIGHT_RECENCY   = w.get('recency',   self.WEIGHT_RECENCY)
        self.WEIGHT_GAP       = w.get('gap',       self.WEIGHT_GAP)
        self.WEIGHT_PAIR      = w.get('pair',      self.WEIGHT_PAIR)
        self.WEIGHT_MOMENTUM  = w.get('momentum',  self.WEIGHT_MOMENTUM)
        self.WEIGHT_ZONE      = w.get('zone',      self.WEIGHT_ZONE)


# =============================================================================
# 2. DataEngine
# =============================================================================
class DataEngine:

    def __init__(self, csv_path: str, config: Config = None,
                 exclude_last_n: int = 0):
        self.csv_path       = csv_path
        self.config         = config or Config()
        self.exclude_last_n = exclude_last_n
        self.is_loaded      = False
        self.error_msg      = ""
        try:
            self._load_data()
            self._analyze_all()
            self.is_loaded = True
        except Exception as e:
            self.error_msg = str(e)

    @classmethod
    def from_history(cls, history: List[List[int]],
                     config: Config = None) -> 'DataEngine':
        obj               = cls.__new__(cls)
        obj.csv_path      = None
        obj.config        = config or Config()
        obj.exclude_last_n = 0
        obj.is_loaded     = False
        obj.error_msg     = ""
        try:
            obj.history         = [row[:] for row in history]
            obj.hidden_draws    = []
            obj.total_rounds    = len(obj.history)
            obj.last_draw       = obj.history[-1] if obj.history else []
            obj.historical_sets = [frozenset(row) for row in obj.history]
            obj._analyze_all()
            obj.is_loaded = True
        except Exception as e:
            obj.error_msg = str(e)
        return obj

    def _load_data(self):
        all_history = load_history(self.csv_path, verbose=False)
        if self.exclude_last_n > 0:
            self.history      = all_history[:-self.exclude_last_n]
            self.hidden_draws = all_history[-self.exclude_last_n:]
        else:
            self.history      = all_history
            self.hidden_draws = []
        self.total_rounds    = len(self.history)
        self.last_draw       = self.history[-1] if self.history else []
        self.historical_sets = [frozenset(row) for row in self.history]

    def _analyze_all(self):
        self.frequency_analysis = self._analyze_frequency()
        self.gap_analysis       = self._analyze_gaps()
        self.pair_analysis      = self._analyze_pairs()
        self.momentum_analysis  = self._analyze_momentum()
        self.zone_analysis      = self._analyze_zones()
        self.pattern_stats      = self._analyze_patterns()
        self.final_scores       = self._calculate_final_scores()

    @staticmethod
    def _robust_normalize(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 46:
            s = arr[1:]
        else:
            s = arr
        if len(s) == 0 or s.max() == s.min():
            return arr.copy()
        median   = np.median(s)
        q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
        iqr      = q75 - q25
        if iqr > 0:
            normalized = (s - median) / iqr
            normalized = np.clip(normalized, -2, 2)
            normalized = (normalized + 2) / 4
        else:
            normalized = s / s.max() if s.max() > 0 else s.copy()
        if len(arr) == 46:
            result     = arr.copy()
            result[1:] = normalized
            return result
        return normalized.copy()

    def _analyze_frequency(self) -> Dict:
        result         = {'total': Counter(), 'by_period': {},
                          'weighted': np.zeros(46)}
        all_nums       = [n for row in self.history for n in row]
        result['total'] = Counter(all_nums)
        period_weights = {10: 4.0, 30: 2.5, 50: 1.5, 100: 1.0}
        for period in self.config.RECENT_PERIODS:
            if period <= self.total_rounds:
                result['by_period'][period] = Counter(
                    n for row in self.history[-period:] for n in row)
        for num in range(1, 46):
            score = 0
            for period, weight in period_weights.items():
                if period in result['by_period']:
                    expected = (period * 6) / 45
                    actual   = result['by_period'][period].get(num, 0)
                    score   += (actual / expected) * weight if expected > 0 else 0
            result['weighted'][num] = score
        result['weighted'] = self._robust_normalize(result['weighted'])
        return result

    def _analyze_gaps(self) -> Dict:
        result      = {'gap': {}, 'avg_gap': {}, 'score': np.zeros(46)}
        appearances = defaultdict(list)
        for idx, row in enumerate(self.history):
            for n in row:
                appearances[n].append(idx)
        for num in range(1, 46):
            if num in appearances and len(appearances[num]) > 1:
                gaps = [appearances[num][i+1] - appearances[num][i]
                        for i in range(len(appearances[num]) - 1)]
                result['avg_gap'][num] = np.mean(gaps)
                result['gap'][num]     = self.total_rounds - 1 - appearances[num][-1]
                ratio = (result['gap'][num] / result['avg_gap'][num]
                         if result['avg_gap'][num] > 0 else 1)
                if 0.8 <= ratio <= 1.5:
                    result['score'][num] = 1.0 - abs(ratio - 1.0) * 0.3
                elif ratio > 1.5:
                    result['score'][num] = max(0.15, 1.0 - (ratio - 1.5) * 0.2)
                else:
                    result['score'][num] = max(0.25, ratio * 0.6)
            else:
                result['gap'][num]     = self.total_rounds
                result['avg_gap'][num] = self.total_rounds / 2
                result['score'][num]   = 0.3
        result['score'] = self._robust_normalize(result['score'])
        return result

    def _analyze_pairs(self) -> Dict:
        result = {
            'pair_count':           Counter(),
            'pair_score':           defaultdict(lambda: defaultdict(float)),
            'top_pairs':            [],
            'number_pair_strength': np.zeros(46),
        }
        window = min(100, self.total_rounds)
        for row in self.history[-window:]:
            for pair in combinations(sorted(row), 2):
                result['pair_count'][pair] += 1
        result['top_pairs'] = result['pair_count'].most_common(50)
        for (a, b), count in result['pair_count'].items():
            strength = count / window
            result['pair_score'][a][b] = strength
            result['pair_score'][b][a] = strength
            result['number_pair_strength'][a] += strength
            result['number_pair_strength'][b] += strength
        result['number_pair_strength'] = self._robust_normalize(
            result['number_pair_strength'])
        return result

    def _analyze_momentum(self) -> Dict:
        result = {'momentum': np.zeros(46), 'trend': {}}
        if self.total_rounds >= 30:
            recent_10 = Counter(n for row in self.history[-10:] for n in row)
            recent_30 = Counter(n for row in self.history[-30:] for n in row)
            for num in range(1, 46):
                short = recent_10.get(num, 0) / 10
                mid   = recent_30.get(num, 0) / 30
                if mid > 0:
                    momentum = short / mid
                    result['momentum'][num] = momentum
                    result['trend'][num] = (
                        'UP' if momentum > 1.3 else
                        'DOWN' if momentum < 0.7 else 'STABLE')
                else:
                    result['momentum'][num] = 1.0 if short > 0 else 0.5
                    result['trend'][num]    = 'UP' if short > 0 else 'DORMANT'
        result['momentum'] = self._robust_normalize(result['momentum'])
        return result

    def _analyze_zones(self) -> Dict:
        zones  = [(1,10),(11,20),(21,30),(31,40),(41,45)]
        result = {'zone_freq': {z: Counter() for z in zones},
                  'zone_score': np.zeros(46)}
        recent = self.history[-50:] if self.total_rounds >= 50 else self.history
        for row in recent:
            for n in row:
                for z_s, z_e in zones:
                    if z_s <= n <= z_e:
                        result['zone_freq'][(z_s, z_e)][n] += 1
                        break
        for z_s, z_e in zones:
            nums   = list(range(z_s, z_e + 1))
            counts = [result['zone_freq'][(z_s, z_e)].get(n, 0) for n in nums]
            mx     = max(counts) if counts else 1
            for i, n in enumerate(nums):
                result['zone_score'][n] = counts[i] / mx if mx > 0 else 0
        return result

    def _analyze_patterns(self) -> Dict:
        result = {'sum_dist': [], 'ac_dist': []}
        for row in self.history:
            nums  = sorted(row)
            diffs = set(abs(nums[j]-nums[i])
                        for i in range(6) for j in range(i+1, 6))
            result['sum_dist'].append(sum(nums))
            result['ac_dist'].append(len(diffs) - 5)
        return result

    def _calculate_final_scores(self) -> np.ndarray:
        scores      = np.zeros(46)
        cfg         = self.config
        recent_10   = self.frequency_analysis['by_period'].get(10, Counter())
        expected_10 = (10 * 6) / 45
        for num in range(1, 46):
            actual  = recent_10.get(num, 0)
            recency = min(actual / expected_10, 1.5) / 1.5
            scores[num] = (
                self.frequency_analysis['weighted'][num]          * cfg.WEIGHT_FREQUENCY
                + recency                                          * cfg.WEIGHT_RECENCY
                + self.gap_analysis['score'][num]                 * cfg.WEIGHT_GAP
                + self.pair_analysis['number_pair_strength'][num] * cfg.WEIGHT_PAIR
                + self.momentum_analysis['momentum'][num]         * cfg.WEIGHT_MOMENTUM
                + self.zone_analysis['zone_score'][num]           * cfg.WEIGHT_ZONE
            )
        s = scores[1:]
        if s.sum() > 0:
            scores[1:] = s / s.sum()
        return scores

    def get_hot_numbers(self, n=10):
        return sorted([(i, self.final_scores[i]) for i in range(1, 46)],
                      key=lambda x: x[1], reverse=True)[:n]

    def get_cold_numbers(self, n=10):
        return sorted([(i, self.final_scores[i]) for i in range(1, 46)],
                      key=lambda x: x[1])[:n]


# =============================================================================
# 3. GeometricAnalyzer
# =============================================================================
class GeometricAnalyzer:

    def __init__(self, engine: DataEngine, n_clusters: int = 5):
        self.engine        = engine
        self.n_clusters    = n_clusters
        self.cluster_stats = {}
        self.top_clusters  = []
        if HAS_SKLEARN and len(engine.history) > 30:
            self._build_clusters()
        else:
            self._build_simple_clusters()

    def _extract_features(self, nums: List[int]) -> np.ndarray:
        nums = sorted(nums)
        f    = []
        f.extend([sum(nums), np.mean(nums), np.std(nums), max(nums)-min(nums)])
        gaps = [nums[i+1]-nums[i] for i in range(5)]
        f.extend([np.mean(gaps), np.std(gaps), max(gaps), min(gaps)])
        f.extend([nums[0], nums[5], nums[2]+nums[3]])
        zones = [0]*5
        for n in nums:
            zones[min((n-1)//10, 4)] += 1
        f.extend(zones)
        diffs = set(abs(nums[j]-nums[i])
                    for i in range(6) for j in range(i+1, 6))
        f.extend([
            sum(1 for n in nums if n%2==1),
            sum(1 for n in nums if n>=23),
            len(diffs)-5,
            len(set(n%10 for n in nums)),
            sum(1 for n in nums if n in PRIME_NUMBERS),
            sum(1 for i in range(5) if nums[i+1]==nums[i]+1),
        ])
        return np.array(f, dtype=float)

    def _build_clusters(self):
        all_features        = [self._extract_features(r) for r in self.engine.history]
        self.feature_matrix = np.array(all_features)
        self.scaler         = StandardScaler()
        self.scaled         = self.scaler.fit_transform(self.feature_matrix)
        best_score, best_k  = -1, self.n_clusters
        max_k = max(3, min(10, len(self.engine.history) // 15))
        for k in range(3, max_k+1):
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.scaled)
            if len(set(labels)) < 2:
                continue
            sc = silhouette_score(self.scaled, labels)
            if sc > best_score:
                best_score, best_k = sc, k
        self.n_clusters = best_k
        self.kmeans     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.labels     = self.kmeans.fit_predict(self.scaled)
        self._compute_cluster_stats()

    def _build_simple_clusters(self):
        self.kmeans     = None
        self.scaler     = None
        self.n_clusters = 3
        sums = [sum(row) for row in self.engine.history]
        q33  = np.percentile(sums, 33)
        q66  = np.percentile(sums, 66)
        self.labels = np.array([
            0 if s < q33 else (1 if s < q66 else 2) for s in sums])
        self._compute_cluster_stats()

    def _compute_cluster_stats(self):
        for cid in range(self.n_clusters):
            mask     = self.labels == cid
            draws    = [self.engine.history[i]
                        for i in range(len(self.engine.history)) if mask[i]]
            recent_n = min(50, len(self.engine.history))
            recent_c = sum(1 for i in range(-recent_n, 0)
                           if (i+len(self.labels)) >= 0
                           and self.labels[i] == cid)
            self.cluster_stats[cid] = {
                'count':        len(draws),
                'ratio':        len(draws) / max(len(self.engine.history), 1),
                'avg_sum':      np.mean([sum(r) for r in draws]) if draws else 0,
                'avg_spread':   np.mean([max(r)-min(r) for r in draws]) if draws else 0,
                'recent_ratio': recent_c / recent_n if recent_n > 0 else 0,
            }
        scored = sorted(
            [(cid, st['ratio']*0.5 + st['recent_ratio']*0.5)
             for cid, st in self.cluster_stats.items()],
            key=lambda x: x[1], reverse=True)
        self.top_clusters = [cid for cid, _ in scored[:3]]

    def get_cluster_id(self, nums: List[int]) -> int:
        if self.kmeans is not None:
            scaled = self.scaler.transform([self._extract_features(nums)])
            return int(self.kmeans.predict(scaled)[0])
        s    = sum(nums)
        sums = [sum(r) for r in self.engine.history]
        q33  = np.percentile(sums, 33)
        q66  = np.percentile(sums, 66)
        return 0 if s < q33 else (1 if s < q66 else 2)

    def get_cluster_bonus(self, nums: List[int]) -> float:
        cid = self.get_cluster_id(nums)
        if cid == self.top_clusters[0]:  return 1.0
        if cid in self.top_clusters[1:]: return 0.6
        return 0.2


# =============================================================================
# 4. FilterSystem
# =============================================================================
class FilterSystem:

    def __init__(self, engine: DataEngine, config: Config = None):
        self.engine          = engine
        self.config          = config or engine.config
        self.rejection_stats = Counter()
        self.total_checked   = 0
        self.total_passed    = 0

    def reset(self):
        self.rejection_stats = Counter()
        self.total_checked   = 0
        self.total_passed    = 0

    def apply_all(self, nums: List[int],
                  track: bool = True) -> Tuple[bool, str]:
        nums   = sorted(nums)
        checks = [
            (self._f_sum,     "합계"),
            (self._f_ac,      "AC값"),
            (self._f_odd,     "홀짝"),
            (self._f_high,    "고저"),
            (self._f_zones,   "구간"),
            (self._f_consec,  "연속"),
            (self._f_endings, "끝수"),
            (self._f_primes,  "소수"),
            (self._f_prev,    "이전회차"),
            (self._f_hist,    "역대중복"),
            (self._f_spread,  "간격"),
            (self._f_edge,    "경계"),
        ]
        if track:
            self.total_checked += 1
        for fn, name in checks:
            if not fn(nums):
                if track:
                    self.rejection_stats[name] += 1
                return False, name
        if track:
            self.total_passed += 1
        return True, ""

    def get_pass_rate(self) -> float:
        return (self.total_passed / self.total_checked
                if self.total_checked > 0 else 0.0)

    def get_rejection_summary(self, top_n: int = 6) -> List[Tuple[str, int, float]]:
        total_rej = sum(self.rejection_stats.values())
        return [
            (name, cnt, cnt/total_rej*100 if total_rej > 0 else 0.0)
            for name, cnt in self.rejection_stats.most_common(top_n)
        ]

    def _f_sum(self, n):
        return self.config.SUM_RANGE[0] <= sum(n) <= self.config.SUM_RANGE[1]

    def _f_ac(self, n):
        d = set(abs(n[j]-n[i]) for i in range(6) for j in range(i+1, 6))
        return (len(d)-5) >= self.config.MIN_AC

    def _f_odd(self, n):
        return sum(1 for x in n if x%2==1) in self.config.ODD_RATES

    def _f_high(self, n):
        return sum(1 for x in n if x>=23) in self.config.HIGH_RATES

    def _f_zones(self, n):
        zc = [0]*5
        for x in n:
            zc[min((x-1)//10, 4)] += 1
        return (sum(1 for c in zc if c>0) >= self.config.MIN_ZONES
                and max(zc) <= self.config.MAX_PER_ZONE)

    def _f_consec(self, n):
        run = 1
        for i in range(5):
            if n[i+1] == n[i]+1:
                run += 1
                if run > self.config.MAX_CONSECUTIVE:
                    return False
            else:
                run = 1
        return True

    def _f_endings(self, n):
        endings = [x%10 for x in n]
        return (len(set(endings)) >= self.config.MIN_UNIQUE_ENDINGS
                and max(Counter(endings).values()) <= self.config.MAX_SAME_ENDING)

    def _f_primes(self, n):
        pc = sum(1 for x in n if x in PRIME_NUMBERS)
        return self.config.PRIME_RANGE[0] <= pc <= self.config.PRIME_RANGE[1]

    def _f_prev(self, n):
        if not self.engine.last_draw:
            return True
        mc = len(set(n) & set(self.engine.last_draw))
        return self.config.PREV_MATCH_RANGE[0] <= mc <= self.config.PREV_MATCH_RANGE[1]

    def _f_hist(self, n):
        return frozenset(n) not in self.engine.historical_sets

    def _f_spread(self, n):
        return 3.5 <= np.mean([n[i+1]-n[i] for i in range(5)]) <= 13.0

    def _f_edge(self, n):
        if 1 in n and 45 in n:
            return sum(1 for x in n if 18<=x<=27) >= 1
        return True


# =============================================================================
# 5. EnsembleGenerator
# =============================================================================
class EnsembleGenerator:
    """
    [v3.4 핵심 수정]

    클러스터 다양성 문제를 2단계로 근본 해결

    [1단계] _algo_multi_cluster(): 비례 확률 → 균등 확률
      - 원본: top_clusters 비율(C1=46%)에 비례 → C1 후보 압도적 생성
      - 수정: 1/len(targets) 균등 확률 → 각 클러스터 동등 기회

    [2단계] _build_diverse_pool(): 클러스터 쿼터 pool 신규 추가
      - 원본: 점수 상위 50개 pool → C1이 점수도 높아 pool도 독점
      - 수정: 클러스터별 최소 count개 우선 확보 후 나머지 점수순 채움
              → pool 구성 단계부터 다양성 보장
    """

    def __init__(self, engine: DataEngine,
                 geometric: GeometricAnalyzer = None):
        self.engine    = engine
        self.geometric = geometric
        self.filter    = FilterSystem(engine)
        self.scores    = engine.final_scores.copy()

        self._total_attempts = 0
        self._total_passed   = 0

    def _get_weights(self) -> np.ndarray:
        w = self.scores[1:].copy()
        return w / w.sum() if w.sum() > 0 else np.ones(45)/45

    # ── _try_add ─────────────────────────────────────────────────────────
    def _try_add(self, nums: List[int], algo: str,
                 results: List[Dict]) -> bool:
        self._total_attempts += 1
        ok, _ = self.filter.apply_all(nums, track=True)
        if ok:
            self._total_passed += 1
            results.append({'nums': nums, 'algo': algo})
            return True
        return False

    # ── 메인 생성 ────────────────────────────────────────────────────────
    def generate(self, count: int = 5,
                 verbose: bool = True) -> List[Dict]:

        # 매 생성마다 필터 통계 초기화
        self.filter.reset()
        self._total_attempts = 0
        self._total_passed   = 0

        all_cands = []
        if verbose:
            print(f"  {C.DM}후보 생성 중...{C.E}")

        c1 = self._algo_weighted(count * 4)
        all_cands.extend(c1)
        if verbose:
            print(f"    Algo 1 (Weighted):     {len(c1):3d} 후보")

        c2 = self._algo_balanced(count * 4)
        all_cands.extend(c2)
        if verbose:
            print(f"    Algo 2 (Balanced):     {len(c2):3d} 후보")

        c3 = self._algo_pattern(count * 4)
        all_cands.extend(c3)
        if verbose:
            print(f"    Algo 3 (Pattern):      {len(c3):3d} 후보")

        if self.geometric:
            c4 = self._algo_multi_cluster(count * 4)
            all_cands.extend(c4)
            if verbose:
                print(f"    Algo 4 (MultiCluster): {len(c4):3d} 후보")

        pass_rate = (self._total_passed / self._total_attempts * 100
                     if self._total_attempts > 0 else 0.0)

        if verbose:
            print(f"  {C.DM}총 {len(all_cands)}개 후보 "
                  f"(시도 {self._total_attempts}회 / "
                  f"통과율 {pass_rate:.1f}%) → 스코어링 중...{C.E}")

        if not all_cands:
            print(f"  {C.R}[경고] 후보 없음. 필터 조건을 완화하세요.{C.E}")
            return []

        scored = self._score_candidates(all_cands)
        final  = self._select_with_coverage(scored, count)

        if verbose:
            print(f"  {C.G}✓ {len(final)}세트 최종 선정 "
                  f"(커버리지 최적화 / 생성 통과율: {pass_rate:.1f}%){C.E}")

        return final

    # ── 알고리즘 1: 가중치 기반 ──────────────────────────────────────────
    def _algo_weighted(self, count: int) -> List[Dict]:
        results  = []
        w        = self._get_weights()
        max_iter = min(count * 20, 80000)
        for _ in range(max_iter):
            if len(results) >= count:
                break
            try:
                nums = sorted(int(n) for n in
                              np.random.choice(range(1,46), 6,
                                               replace=False, p=w))
                self._try_add(nums, 'weighted', results)
            except Exception:
                continue
        return results

    # ── 알고리즘 2: 구간 밸런스 ──────────────────────────────────────────
    def _algo_balanced(self, count: int) -> List[Dict]:
        results  = []
        zones    = [(1,10),(11,20),(21,30),(31,40),(41,45)]
        w        = self._get_weights()
        max_iter = min(count * 20, 80000)
        for _ in range(max_iter):
            if len(results) >= count:
                break
            picks = [1,1,1,1,1]
            picks[np.random.randint(0,5)] += 1
            nums  = []
            valid = True
            for i, (z_s, z_e) in enumerate(zones):
                zone_nums = list(range(z_s, z_e+1))
                zone_w    = np.array([w[n-1] for n in zone_nums])
                zs        = zone_w.sum()
                zone_w    = (zone_w/zs if zs>0
                             else np.ones(len(zone_nums))/len(zone_nums))
                try:
                    picked = np.random.choice(zone_nums, picks[i],
                                              replace=False, p=zone_w)
                    nums.extend(int(n) for n in picked)
                except Exception:
                    valid = False
                    break
            if not valid:
                continue
            nums = sorted(set(nums))
            if len(nums) == 6:
                self._try_add(nums, 'balanced', results)
        return results

    # ── 알고리즘 3: 동반출현 패턴 ────────────────────────────────────────
    def _algo_pattern(self, count: int) -> List[Dict]:
        results   = []
        top_pairs = self.engine.pair_analysis['top_pairs'][:30]
        w         = self._get_weights()
        max_iter  = min(count * 20, 80000)
        for _ in range(max_iter):
            if len(results) >= count:
                break
            if top_pairs and np.random.random() < 0.7:
                idx  = np.random.randint(0, min(15, len(top_pairs)))
                base = list(top_pairs[idx][0])
            else:
                base = []
            nums_set = set(base)
            needed   = 6 - len(nums_set)
            temp_w   = w.copy()
            for n in nums_set:
                temp_w[n-1] = 0.0
            ts = temp_w.sum()
            if ts <= 0:
                continue
            temp_w /= ts
            try:
                extra = np.random.choice(range(1,46), needed,
                                         replace=False, p=temp_w)
                nums_set.update(int(n) for n in extra)
            except Exception:
                continue
            if len(nums_set) == 6:
                self._try_add(sorted(nums_set), 'pattern', results)
        return results

    # ── 알고리즘 4: 다중 클러스터 ────────────────────────────────────────
    def _algo_multi_cluster(self, count: int) -> List[Dict]:
        """
        [v3.4 핵심 수정 1단계] 비례 확률 → 균등 확률
        원본: probs = [ratio/total_r ...]  → C1(46%)이 절반 차지
        수정: probs = [1/n, 1/n, 1/n]     → C1/C2/C0 동등 기회
        → C2(저합계 108), C0(고합계 171) 후보가 pool에 충분히 생성됨
        """
        results  = []
        w        = self._get_weights()
        max_iter = min(count * 25, 80000)

        targets = [
            {
                'cluster':       cid,
                'target_sum':    int(self.geometric.cluster_stats[cid]['avg_sum']),
                'target_spread': int(self.geometric.cluster_stats[cid]['avg_spread']),
            }
            for cid in self.geometric.top_clusters
        ]
        if not targets:
            return results

        # [수정] 균등 확률 — 비율 무관하게 각 클러스터 동등 기회
        probs = [1.0 / len(targets)] * len(targets)

        for _ in range(max_iter):
            if len(results) >= count:
                break

            chosen        = targets[np.random.choice(len(targets), p=probs)]
            target_spread = max(5, chosen['target_spread'])
            first = np.random.randint(1, max(2, 46-target_spread))
            last  = min(45, first + target_spread + np.random.randint(-5, 6))
            last  = max(first+5, last)
            seeds = list({first, min(45, last)})

            temp_w = w.copy()
            for n in seeds:
                if 1 <= n <= 45:
                    temp_w[n-1] = 0.0
            ts = temp_w.sum()
            if ts <= 0:
                continue
            temp_w /= ts

            try:
                needed   = 6 - len(seeds)
                extra    = np.random.choice(range(1,46), needed,
                                            replace=False, p=temp_w)
                all_nums = set(seeds)
                all_nums.update(int(n) for n in extra)
            except Exception:
                continue

            if len(all_nums) < 6:
                continue
            if len(all_nums) > 6:
                all_nums = set(sorted(all_nums,
                                      key=lambda n: self.scores[n],
                                      reverse=True)[:6])
            nums = sorted(all_nums)
            if abs(sum(nums) - chosen['target_sum']) <= 35:
                self._try_add(nums, f"cluster_{chosen['cluster']}", results)

        return results

    # ── 스코어링 ─────────────────────────────────────────────────────────
    def _score_candidates(self, candidates: List[Dict]) -> List[Dict]:
        seen            = set()
        unique          = []
        raw_num_scores  = []
        raw_pair_scores = []

        for cand in candidates:
            key = tuple(cand['nums'])
            if key in seen:
                continue
            seen.add(key)
            nums = cand['nums']
            raw_num_scores.append(sum(self.scores[n] for n in nums))
            raw_pair_scores.append(
                sum(self.engine.pair_analysis['pair_score'][a][b]
                    for a, b in combinations(nums, 2)))
            unique.append(cand)

        if not unique:
            return []

        max_num  = max(raw_num_scores)  if max(raw_num_scores)  > 0 else 1.0
        max_pair = max(raw_pair_scores) if max(raw_pair_scores) > 0 else 1.0

        scored = []
        for i, cand in enumerate(unique):
            nums  = cand['nums']
            s     = sum(nums)
            diffs = set(abs(nums[j]-nums[i])
                        for i in range(6) for j in range(i+1, 6))
            ac    = len(diffs) - 5
            zones = [0]*5
            for n in nums:
                zones[min((n-1)//10, 4)] += 1

            sum_score     = 1.0 if 110 <= s <= 190 else 0.5
            ac_score      = np.clip(ac / 10.0,  0.0, 1.0)
            num_score     = np.clip(raw_num_scores[i]  / max_num,  0.0, 1.0)
            pair_score    = np.clip(raw_pair_scores[i] / max_pair, 0.0, 1.0)
            balance_score = np.clip(1.0 / (1.0 + np.var(zones)), 0.0, 1.0)
            cluster_bonus = (self.geometric.get_cluster_bonus(nums)
                             if self.geometric else 0.5)
            algo_bonus    = 1.0 if cand['algo'] == 'balanced' else 0.8

            total = (
                sum_score     * 0.12
                + ac_score    * 0.12
                + num_score   * 0.28
                + pair_score  * 0.18
                + balance_score * 0.13
                + cluster_bonus * self.engine.config.CLUSTER_BONUS_WEIGHT
                + algo_bonus  * 0.07
            )

            cand['stat'] = {
                'sum':           s,
                'ac':            ac,
                'odd':           sum(1 for n in nums if n%2==1),
                'high':          sum(1 for n in nums if n>=23),
                'prime':         sum(1 for n in nums if n in PRIME_NUMBERS),
                'cluster_bonus': cluster_bonus,
                'cluster_id':    (self.geometric.get_cluster_id(nums)
                                  if self.geometric else -1),
            }
            cand['total_score'] = round(total, 6)
            scored.append(cand)

        return scored

    # ── [v3.4 신규] 클러스터 쿼터 pool 구성 ─────────────────────────────
    def _build_diverse_pool(self, candidates: List[Dict],
                            count: int) -> List[Dict]:
        """
        [v3.4 핵심 수정 2단계] 클러스터 쿼터 pool
        원본: 점수 상위 50개 → C1이 점수도 높아 pool 독점
        수정:
          Step 1. 클러스터별로 점수순 정렬
          Step 2. 각 클러스터에서 최소 count개 우선 확보
          Step 3. 남은 슬롯은 전체 점수순으로 채움
        → pool 구성 단계부터 C2/C0 클러스터 후보 포함 보장
        """
        pool_size = max(count * 10, 50)

        # 클러스터별 점수순 분류
        by_cluster: Dict[int, List[Dict]] = defaultdict(list)
        for cand in sorted(candidates,
                           key=lambda x: x['total_score'], reverse=True):
            cid = cand['stat'].get('cluster_id', -1)
            by_cluster[cid].append(cand)

        pool     = []
        seen_ids = set()

        # Step 1: 각 클러스터에서 최소 count개 우선 확보
        for cid, cands in by_cluster.items():
            for c in cands[:count]:
                if id(c) not in seen_ids:
                    pool.append(c)
                    seen_ids.add(id(c))

        # Step 2: 나머지 슬롯을 전체 점수순으로 채움
        all_sorted = sorted(candidates,
                            key=lambda x: x['total_score'], reverse=True)
        for c in all_sorted:
            if len(pool) >= pool_size:
                break
            if id(c) not in seen_ids:
                pool.append(c)
                seen_ids.add(id(c))

        return pool

    # ── 커버리지 + 클러스터 다양성 최적화 선택 ───────────────────────────
    def _select_with_coverage(self, candidates: List[Dict],
                              count: int) -> List[Dict]:
        """
        [v3.4 수정]
        pool 구성: sorted[:50] → _build_diverse_pool() 호출
        max_same_cls: max(1, count//2) 동적 계산 유지
        """
        if not candidates:
            return []

        # [수정] 클러스터 쿼터 pool 사용
        pool = self._build_diverse_pool(candidates, count)

        selected        = []
        used_nums       = set()
        cluster_counter = Counter()
        max_overlap     = self.engine.config.MAX_SET_OVERLAP
        max_same_cls    = max(1, count // 2)  # count=5 → 2세트 제한

        for _ in range(count):
            best_idx   = -1
            best_value = -1.0

            for i, cand in enumerate(pool):
                if cand in selected:
                    continue
                nums_set = set(cand['nums'])
                cid      = cand['stat'].get('cluster_id', -1)

                # ① 세트 간 번호 중복 체크
                if not all(len(nums_set & set(s['nums'])) <= max_overlap
                           for s in selected):
                    continue

                # ② 클러스터 다양성 체크
                if cluster_counter[cid] >= max_same_cls:
                    continue

                new_coverage = len(nums_set - used_nums)
                value = cand['total_score'] * 0.7 + (new_coverage / 6.0) * 0.3

                if value > best_value:
                    best_value = value
                    best_idx   = i

            if best_idx >= 0:
                chosen = pool[best_idx]
                selected.append(chosen)
                used_nums.update(chosen['nums'])
                cluster_counter[chosen['stat'].get('cluster_id', -1)] += 1
            else:
                # fallback 1: 클러스터 제한만 해제
                remaining = [c for c in pool
                             if c not in selected
                             and all(len(set(c['nums']) & set(s['nums']))
                                     <= max_overlap for s in selected)]
                if remaining:
                    fb = max(remaining, key=lambda x: x['total_score'])
                    selected.append(fb)
                    used_nums.update(fb['nums'])
                    cluster_counter[fb['stat'].get('cluster_id', -1)] += 1
                elif pool:
                    # fallback 2: 모든 조건 해제
                    left = [c for c in pool if c not in selected]
                    if left:
                        fb = max(left, key=lambda x: x['total_score'])
                        selected.append(fb)
                        used_nums.update(fb['nums'])

        return selected


# =============================================================================
# 6. BacktestEngine
# =============================================================================
class BacktestEngine:

    def __init__(self, csv_path: str = None,
                 config: Config = None,
                 preloaded_history: List[List[int]] = None):
        self.config = config or Config()
        if preloaded_history is not None:
            self.all_hist = preloaded_history
        elif csv_path is not None:
            self.all_hist = load_history(csv_path, verbose=False)
        else:
            raise ValueError("csv_path 또는 preloaded_history 필요")
        self.total = len(self.all_hist)

    def run(self, test_rounds: int = 20,
            preds_per_round: int = 5,
            start_from_end: int = None,
            rand_per_round: int = 100,
            verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n{'='*65}")
            print(f"  🔬 BACKTEST  {test_rounds}회차 × {preds_per_round}세트")
            print(f"{'='*65}\n")

        if start_from_end is not None:
            end_idx   = min(self.total - start_from_end + test_rounds,
                            self.total)
            start_idx = end_idx - test_rounds
        else:
            start_idx = self.total - test_rounds
            end_idx   = self.total

        start_idx     = max(start_idx, 100)
        actual_rounds = end_idx - start_idx

        if actual_rounds <= 0:
            print(f"  {C.R}[오류] 유효 검증 구간 없음{C.E}")
            return {}

        match_dist  = Counter()
        all_matches = []
        highlights  = []

        # [통합 ②] 랜덤 대조군 (v2.3 이식) — 실측 랜덤과 공정 비교
        rand_dist        = Counter()
        rand_all_matches = []
        sys_round_best   = []
        rand_round_best  = []

        for i in range(actual_rounds):
            round_idx = start_idx + i
            actual    = set(self.all_hist[round_idx])
            engine    = DataEngine.from_history(
                            self.all_hist[:round_idx], self.config)
            if not engine.is_loaded:
                continue
            geometric = GeometricAnalyzer(engine, self.config.N_CLUSTERS)
            generator = EnsembleGenerator(engine, geometric)
            preds     = generator.generate(count=preds_per_round, verbose=False)

            round_best = 0
            for pred in preds:
                mc = len(set(pred['nums']) & actual)
                match_dist[mc] += 1
                all_matches.append(mc)
                round_best = max(round_best, mc)
                if mc >= 3:
                    highlights.append((round_idx+1, pred['nums'],
                                       sorted(actual), mc))
            sys_round_best.append(round_best)

            # [개선 ③] 확대된 랜덤 대조군 (분산 축소용, rand_per_round세트)
            #  - 라운드 최대/3+ 비교는 공정성을 위해 앞쪽 len(preds)세트만 사용
            round_rand = []
            for _ in range(max(rand_per_round, len(preds))):
                r_set = set(random.sample(range(1, 46), 6))
                r_mc  = len(r_set & actual)
                rand_dist[r_mc] += 1
                rand_all_matches.append(r_mc)
                round_rand.append(r_mc)
            rand_round_best.append(max(round_rand[:len(preds)]) if preds else 0)
            if verbose and round_best >= 2:
                sym = "🎯" if round_best >= 3 else "  "
                print(f"  {sym} 회차 {round_idx+1:4d}: "
                      f"최대 {round_best}개 일치 (실제: {sorted(actual)})")

        if not all_matches:
            return {}

        avg        = float(np.mean(all_matches))
        random_exp = 6 * 6 / 45
        improvement = ((avg / random_exp) - 1) * 100 if random_exp > 0 else 0.0
        composite  = (avg
                      + match_dist.get(3, 0) * 0.3
                      + match_dist.get(4, 0) * 1.0
                      + match_dist.get(5, 0) * 5.0)

        rand_avg = float(np.mean(rand_all_matches)) if rand_all_matches else 0.0
        result = {
            'avg_match':          avg,
            'rand_avg_match':     rand_avg,
            'rand_distribution':  rand_dist,
            'rand_pool':          rand_all_matches,
            'sys_matches':        all_matches,
            'sys_round_best':     sys_round_best,
            'rand_round_best':    rand_round_best,
            'best_match':         max(all_matches),
            'total_predictions':  len(all_matches),
            'match_distribution': match_dist,
            'improvement_pct':    improvement,
            'composite_score':    composite,
            'highlights':         highlights,
        }
        if verbose:
            self._print_result(result, random_exp)
        return result

    @staticmethod
    def _print_result(result: Dict, random_exp: float):
        total = result['total_predictions']
        avg   = result['avg_match']
        imp   = result['improvement_pct']
        md    = result['match_distribution']
        print(f"\n  {'─'*55}")
        print(f"  📊 백테스트 요약  ({total}세트)")
        print(f"  {'─'*55}")
        print(f"  평균 일치 : {avg:.3f}  (랜덤 기대: {random_exp:.3f})")
        print(f"  최대 일치 : {result['best_match']}개")
        print(f"  복합 점수 : {result['composite_score']:.4f}")
        for mc in sorted(md.keys(), reverse=True):
            cnt = md[mc]
            pct = cnt/total*100
            print(f"    {mc}개: {cnt:4d}회 ({pct:5.1f}%) "
                  f"{'█'*max(1,int(pct/2))}")
        tag = f"+{imp:.1f}%" if imp >= 0 else f"{imp:.1f}%"
        print(f"\n  {'✅' if imp>0 else '⚠️'} 이론적 랜덤 기대 대비 {tag}")

        # [통합 ②] 실측 랜덤 대조군 비교 (v2.3 이식)
        if 'rand_avg_match' in result:
            r_avg   = result['rand_avg_match']
            s_best  = result['sys_round_best']
            r_best  = result['rand_round_best']
            s_mb    = float(np.mean(s_best))  if s_best else 0.0
            r_mb    = float(np.mean(r_best))  if r_best else 0.0
            s_3p    = sum(1 for m in s_best if m >= 3) / max(len(s_best), 1) * 100
            r_3p    = sum(1 for m in r_best if m >= 3) / max(len(r_best), 1) * 100
            print(f"\n  {'─'*55}")
            print(f"  🎲 실측 랜덤 대조군 비교")
            print(f"  {'─'*55}")
            print(f"                      시스템      랜덤(확대 표본)")
            print(f"  세트 평균 일치   : {avg:6.3f}    {r_avg:6.3f}")
            print(f"  라운드 최대 평균 : {s_mb:6.3f}    {r_mb:6.3f}  (동일 세트 수 기준)")
            print(f"  3+ 일치 라운드   : {s_3p:5.1f}%    {r_3p:5.1f}%  (동일 세트 수 기준)")

            # [신규 ②] 부트스트랩 귀무분포 판정 — 임의 문턱 대신 95% 구간
            n_lo, n_hi, pct, verdict = null_distribution_verdict(
                result['sys_matches'], result['rand_pool'])
            c_lo, c_hi = bootstrap_mean_ci(result['sys_matches'])
            print(f"\n  🧪 부트스트랩 판정 (n_boot=2000)")
            print(f"  시스템 평균 95% CI : {c_lo:.3f} ~ {c_hi:.3f}")
            print(f"  랜덤 귀무분포 95%  : {n_lo:.3f} ~ {n_hi:.3f}")
            print(f"  시스템 백분위      : {pct:.1f}번째")
            print(f"  ⚖️  판정: {verdict}")
        if result['highlights']:
            print(f"\n  🎯 3개 이상 일치 하이라이트")
            for rnd, pred, actual, mc in result['highlights'][:5]:
                print(f"    회차 {rnd:4d}: {pred} → {mc}개 (실제:{actual})")


# =============================================================================
# 7. HyperparameterOptimizer
# =============================================================================
class HyperparameterOptimizer:

    def __init__(self, csv_path: str, config: Config = None):
        self.config = config or Config()
        df            = pd.read_csv(csv_path)
        self.all_hist = [[int(n) for n in row]
                         for row in df.iloc[:, 2:8].values]
        self.total    = len(self.all_hist)
        self.history: List[Dict] = []

    def optimize(self, n_trials: int = 20,
                 opt_rounds: int = 10,
                 valid_rounds: int = 10,
                 preds_per_round: int = 5,
                 verbose: bool = True) -> Config:
        if verbose:
            print(f"\n{'='*65}")
            print(f"  🔧 HYPERPARAMETER OPTIMIZATION  {n_trials} Trials")
            print(f"  최적화 구간: 뒤에서 {valid_rounds+opt_rounds}~{valid_rounds}회차")
            print(f"  검증   구간: 뒤에서 {valid_rounds}~0회차  (분리됨)")
            print(f"{'='*65}\n")

        weight_names = ['frequency','recency','gap','pair','momentum','zone']
        best_score   = -1.0
        best_weights = self.config.get_weights_dict()

        for trial in range(n_trials):
            raw           = np.random.dirichlet(np.ones(6))
            trial_weights = {name: float(raw[idx])
                             for idx, name in enumerate(weight_names)}
            trial_config  = deepcopy(self.config)
            trial_config.set_weights_from_dict(trial_weights)

            bt = BacktestEngine(preloaded_history=self.all_hist,
                                config=trial_config)
            result = bt.run(
                test_rounds     = opt_rounds,
                preds_per_round = preds_per_round,
                start_from_end  = valid_rounds + opt_rounds,
                verbose         = False)
            if not result:
                continue

            score = result.get('composite_score', 0.0)
            self.history.append({
                'trial':   trial+1,
                'weights': trial_weights,
                'score':   score,
                'avg':     result.get('avg_match', 0.0),
            })
            if score > best_score:
                best_score   = score
                best_weights = trial_weights.copy()

            if verbose:
                marker = f"{C.Y}★{C.E}" if score == best_score else " "
                top_w  = max(trial_weights.items(), key=lambda x: x[1])
                print(f"  {marker} Trial {trial+1:3d}: "
                      f"score={score:.4f}  avg={result['avg_match']:.3f}  "
                      f"top={top_w[0]}({top_w[1]:.2f})")

        if verbose:
            print(f"\n  🏆 최적 가중치:")
            for name, val in best_weights.items():
                print(f"    {name:<12}: {val:.3f}  {'█'*int(val*30)}")
            print(f"  최적 복합 점수: {best_score:.4f}\n")

        optimal = deepcopy(self.config)
        optimal.set_weights_from_dict(best_weights)
        return optimal


# =============================================================================
# 8. 대시보드 출력
# =============================================================================
def print_dashboard(engine: DataEngine,
                    results: List[Dict],
                    geometric: GeometricAnalyzer = None,
                    generator: 'EnsembleGenerator' = None):

    print(f"\n{C.H}")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║      🎰 LOTTO ANALYTICS ULTIMATE v3.4 — Enhanced Edition    ║")
    print("║   Soft Clustering + Coverage Opt + Cluster Diversity v2     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"{C.E}")

    print(f"{C.BD}━━━ 📊 DATA SUMMARY ━━━{C.E}")
    print(f"  총 {engine.total_rounds}회차  |  마지막 당첨: {sorted(engine.last_draw)}")

    if geometric:
        print(f"\n{C.BD}━━━ 🔮 CLUSTER ANALYSIS ━━━{C.E}")
        print(f"  {geometric.n_clusters}개 클러스터  |  "
              f"상위 타겟: {geometric.top_clusters}")
        for cid in geometric.top_clusters:
            st = geometric.cluster_stats[cid]
            print(f"    C{cid}: {st['count']:3d}회 ({st['ratio']*100:4.1f}%)  "
                  f"평균합:{st['avg_sum']:5.0f}  최근:{st['recent_ratio']*100:4.1f}%")

    print(f"\n{C.BD}━━━ 🔥 HOT / ❄️ COLD ━━━{C.E}")
    hot  = engine.get_hot_numbers(8)
    cold = engine.get_cold_numbers(5)
    print(f"  HOT : {' '.join(f'{C.R}{n:2d}{C.E}' for n,_ in hot)}")
    print(f"  COLD: {' '.join(f'{C.B}{n:2d}{C.E}' for n,_ in cold)}")

    print(f"\n{C.BD}━━━ 🎯 PREDICTION RESULTS ━━━{C.E}")
    print(f"  {'#':<3}│{'번호 조합':<26}│{'합':>4}│{'AC':>3}│"
          f"{'홀':>2}│{'소수':>3}│{'CID':>4}│{'Cls':>5}│{'점수':>8}│Algo")
    print(f"  {'─'*80}")
    for i, res in enumerate(results, 1):
        n   = res['nums']
        st  = res['stat']
        ns  = " ".join(f"{x:2d}" for x in n)
        cid = st.get('cluster_id', -1)
        cb  = st.get('cluster_bonus', 0.0)
        print(f"  {i:<3}│{C.G}{ns:<26}{C.E}│{st['sum']:4d}│"
              f"{st['ac']:3d}│{st['odd']:2d}│{st['prime']:3d}│"
              f"{cid:4d}│{cb:5.2f}│{res['total_score']:8.5f}│{res['algo']}")
    print(f"  {'─'*80}")

    all_nums = set(n for r in results for n in r['nums'])
    cov      = len(all_nums)
    print(f"\n  📈 커버리지: {cov}개 / 45개 ({cov/45*100:.1f}%)")

    if geometric:
        cls_dist = Counter(r['stat'].get('cluster_id', -1) for r in results)
        print(f"  🔀 클러스터 분포: "
              + "  ".join(f"C{cid}:{cnt}세트"
                          for cid, cnt in sorted(cls_dist.items())))

    print(f"\n  🔁 세트 간 중복 현황:")
    has_overlap = False
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            ov = set(results[i]['nums']) & set(results[j]['nums'])
            if ov:
                print(f"    세트 {i+1}↔{j+1}: {sorted(ov)} ({len(ov)}개)")
                has_overlap = True
    if not has_overlap:
        print(f"    (중복 없음)")

    if generator and generator.filter.rejection_stats:
        print(f"\n{C.DM}━━━ 🔍 Filter Rejection Stats ━━━{C.E}")
        total_chk  = generator.filter.total_checked
        total_pass = generator.filter.total_passed
        print(f"  전체 시도: {total_chk}회  통과: {total_pass}회  "
              f"통과율: {generator.filter.get_pass_rate()*100:.1f}%")
        for name, cnt, pct in generator.filter.get_rejection_summary(6):
            bar = "█" * max(1, int(pct/3))
            print(f"  {name:<8}: {cnt:5d}회 ({pct:5.1f}%) {bar}")

    print(f"\n{C.CN}━━━ 📋 COPY FORMAT ━━━{C.E}")
    for i, res in enumerate(results, 1):
        print(f"{', '.join(str(n) for n in res['nums'])}")
    print()


# =============================================================================
# 9. main
# =============================================================================
def main(csv_path: str, count: int = 5,
         run_backtest: bool = False,
         run_optimize: bool = False,
         seed: int = 42,
         track: bool = True,
         log_path: str = None):

    np.random.seed(seed)
    random.seed(seed)          # [통합 ③] 랜덤 대조군 재현성

    print(f"\n{C.CN}╔══════════════════════════════════════════════╗{C.E}")
    print(f"{C.CN}║  🎰 LOTTO INTEGRATED v1.1 — Initializing...  ║{C.E}")
    print(f"{C.CN}╚══════════════════════════════════════════════╝{C.E}\n")

    config = Config()

    if run_optimize:
        print(f"{C.Y}[Step 1] Hyperparameter Optimization...{C.E}")
        optimizer = HyperparameterOptimizer(csv_path, config)
        config    = optimizer.optimize(n_trials=20, opt_rounds=10,
                                       valid_rounds=10, preds_per_round=5)
    else:
        print(f"{C.DM}[Step 1] 최적화 스킵 (기본 가중치 사용){C.E}")

    if run_backtest:
        print(f"\n{C.Y}[Step 2] Backtesting...{C.E}")
        all_hist = load_history(csv_path, verbose=True)
        bt = BacktestEngine(preloaded_history=all_hist, config=config)
        bt.run(test_rounds=20, preds_per_round=5, verbose=True)
    else:
        print(f"{C.DM}[Step 2] 백테스트 스킵{C.E}")

    # ── [신규 ①] Step 2.5: 예측 추적 — 지난 예측 자동 대조 + 누적 보고 ──
    tracker      = None
    target_round = None
    if track:
        print(f"\n{C.Y}[Step 2.5] Prediction Tracking...{C.E}")
        rounds, hist_now = load_history_ex(csv_path, verbose=False)
        if log_path is None:
            base     = os.path.dirname(os.path.abspath(csv_path))
            log_path = os.path.join(base, 'prediction_log.csv')
        tracker   = PredictionTracker(log_path)
        round_map = {r: h for r, h in zip(rounds, hist_now) if r is not None}
        tracker.reconcile(round_map)
        tracker.report()
        if round_map:
            target_round = max(round_map.keys()) + 1

    print(f"\n{C.Y}[Step 3] Main Analysis & Prediction...{C.E}")
    engine = DataEngine(csv_path, config)
    if not engine.is_loaded:
        print(f"{C.R}[Error] {engine.error_msg}{C.E}")
        return None, None, None

    print(f"{C.G}  ✓ 데이터 로드 완료 ({engine.total_rounds}회차){C.E}")

    geometric = GeometricAnalyzer(engine, config.N_CLUSTERS)
    print(f"{C.G}  ✓ 클러스터링 완료 "
          f"({geometric.n_clusters}개, 상위: {geometric.top_clusters}){C.E}")

    generator = EnsembleGenerator(engine, geometric)
    results   = generator.generate(count=count, verbose=True)

    if not results:
        print(f"{C.R}[Error] 예측 결과 없음{C.E}")
        return engine, geometric, None

    print_dashboard(engine, results, geometric, generator)

    # ── [신규 ①] 이번 예측을 로그에 사전 등록 ──
    if tracker is not None:
        tracker.log_predictions(target_round, results, seed)

    print(f"{'='*65}")
    print(f"  💡 통계 기반 분석이며 당첨을 보장하지 않습니다.")
    print(f"{'='*65}")
    print(f"  🍀 행운을 빕니다!\n")

    return engine, geometric, results


# =============================================================================
# 편의 함수 & Entry Point
# =============================================================================
def quick_predict(csv_path: str, count: int = 5, seed: int = 42,
                  track: bool = True, log_path: str = None):
    return main(csv_path, count,
                run_backtest=False, run_optimize=False, seed=seed,
                track=track, log_path=log_path)

def full_analysis(csv_path: str, count: int = 5, seed: int = 42,
                  track: bool = True, log_path: str = None):
    return main(csv_path, count,
                run_backtest=True, run_optimize=True, seed=seed,
                track=track, log_path=log_path)


if __name__ == '__main__':
    CSV_PATH = '/content/new_1235.csv'

    # 빠른 예측
    engine, geometric, results = quick_predict(CSV_PATH, count=5)

    # 전체 분석 (시간 소요)
    # engine, geometric, results = full_analysis(CSV_PATH, count=5)