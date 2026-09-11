"""Fresh single-stock Pareto diagnostics, separate from frozen historical accounts.

Always refresh the requested stock before calculation. A single vector cannot
establish full-universe F1: compare only exact-date, verified frozen peers. When
that cross-section is unavailable, report unknown instead of granting a trivial
one-stock F1 or carrying forward an older front. No historical deadline or old
account position is applied to the user's current holding checkbox.
"""
from datetime import date, datetime, time
from zipfile import BadZipFile

import numpy as np

import live_data
import pareto_web
from pareto_strategy import (make_objectives, risk_statistics, risk_target,
                             RiskConfig, STRATEGY_VERSION)
from pareto_backtest import aligned_actions, causal_adjusted_prices, MAINBOARD


_DATA_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError,
                OverflowError, EOFError, BadZipFile)


def _day(value):
    # Validate BEFORE coercing to U10: otherwise timestamps silently lose time.
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('日线日期须为有效的YYYY-MM-DD')
    return value


def _fresh_arrays(fresh):
    """Validate the refresh boundary without substituting any historical rows."""
    try:
        days = [_day(row['day']) for row in fresh['rows']]
        if not days or any(a >= b for a, b in zip(days, days[1:])):
            raise ValueError('empty or unordered history')
        if fresh['latest_date'] != days[-1] or fresh.get('date', days[-1]) != days[-1]:
            raise ValueError('mixed quote dates')
        clock = datetime.fromisoformat(fresh['checked_at'])
        if clock.utcoffset() is None:
            raise ValueError('naive refresh clock')
        clock = clock.astimezone(live_data.SHANGHAI)
        today = clock.date().isoformat()
        if days[-1] > today or (days[-1] == today and clock.time() < time(15)):
            raise ValueError('uncompleted daily bar')
        raw = np.asarray([[row[key] for key in live_data.FIELDS]
                          for row in fresh['rows']])
        if raw.shape != (len(days), 5) or raw.dtype.kind not in 'iuf':
            raise ValueError('invalid numeric rows')
        if any(isinstance(row[key], bool) for row in fresh['rows'] for key in live_data.FIELDS):
            raise ValueError('boolean OHLCV')
        raw = raw.astype(float)
        if (not np.isfinite(raw).all() or np.any(raw[:, :4] <= 0)
                or np.any(raw[:, 4] < 0)
                or np.any(raw[:, 1] < np.maximum(raw[:, 0], raw[:, 3]))
                or np.any(raw[:, 2] > np.minimum(raw[:, 0], raw[:, 3]))):
            raise ValueError('invalid OHLCV')
    except _DATA_ERRORS:
        raise ValueError('本次刷新日线或时间字段无效，暂不下结论；不回落旧数据。') from None
    return np.array(days, dtype='U10'), raw, today


def _verified_actions(actions, code, dates, today):
    """Optional coverage evidence must agree; a false/unknown gate never passes.

    The live provider supplies all coverage fields. Older/minimal adapters may
    omit them, but cannot override explicit contrary evidence with one flag.
    Unverified settlement/tax approximations are NOT coverage-failure flags.
    """
    try:
        meta = actions['metadata']
        if (actions['status'] != 'ok' or actions['rights_issue_present'] is not False
                or meta.get('live_recommendation_blocked') is not False
                or actions.get('code', code) != code):
            return False
        for field in ('coverage_matches_quote', 'pagination_complete'):
            if field in meta and meta[field] is not True:
                return False
        if ('coverage_start' in meta and _day(meta['coverage_start']) > dates[0]
                or 'coverage_end' in meta and not dates[-1] <= _day(meta['coverage_end']) <= today
                or 'quote_date' in meta and _day(meta['quote_date']) != dates[-1]):
            return False
        if not isinstance(actions['events'], list):
            return False
        for event in actions['events']:
            if _day(event['ex_date']) > today:
                return False
            cash, mult = event['cash_per_share'], event['share_multiplier']
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not np.isfinite(v) for v in (cash, mult)) or cash < 0 or mult <= 0:
                return False
    except _DATA_ERRORS + (AttributeError,):
        return False
    return True


def historical_context(code):
    """An absent report must not prevent a fresh quote/indicator request."""
    try:
        folder, manifest, summary, risk, config = pareto_web._load_run()
    except _DATA_ERRORS:
        return dict(folder=None, manifest=None, summary=None, risk=RiskConfig(),
                    risk_source='model_default', account=None, ledger=[],
                    error='历史回测报告缺失或不完整；使用模型默认15%波动目标，不以旧评分替代。')
    context = dict(folder=folder, manifest=manifest, summary=summary, risk=risk,
                   risk_source='frozen_selected', account=None, ledger=[], error=None)
    try:
        if code in {s['code'] for s in manifest['universe']}:
            account = pareto_web._load_account(folder, code, config, manifest['effective_end'])
            ledger = pareto_web._json(folder, f'ledgers/{code}.json')
            if not isinstance(ledger, list):
                raise ValueError('invalid ledger')
            context.update(account=account, ledger=ledger)
        else:
            context['error'] = '当前股票不在冻结股票池；无该股历史账户，不代表代码无效。'
    except _DATA_ERRORS:
        # A missing stock ledger does not erase a valid peer report/selected risk.
        context['error'] = '该股历史账户或账本不完整；保留冻结报告日期及所选风险参数。'
    return context


def compare_same_day(code, day, vector, context):
    """Determine strict dominance against a frozen, exact-date cross-section.

    Fresh requested data may differ from the historical provider, so this is a
    new comparison, never a historical F1 label. Missing peer files make positive
    F1 unknown. One actual dominating peer suffices to reject F1. Eligible peers
    with identical coordinates do not dominate. No HTTP or production DB read.
    """
    manifest = context['manifest']
    result = dict(status='unknown', compared=0, expected=0, coverage='无同日全市场截面',
                  dominators=[], date=day, in_universe=None)
    vector = np.asarray(vector)
    if vector.shape != (34,) or vector.dtype.kind not in 'iuf' or not np.isin(vector, [-1, 0, 1]).all():
        raise ValueError('invalid current vector')
    if not manifest:
        return result
    universe = manifest['universe']
    result['in_universe'] = any(s['code'] == code for s in universe)
    result['expected'] = len(universe) - int(result['in_universe'])
    scope = ('冻结股票池同日有效截面（不是今日全市场实时刷新）' if result['in_universe'] else
             '当前股票不在冻结股票池；仅与冻结池同日有效股票比较，不代表全市场F1')
    if day not in manifest['calendar'] or day > manifest['effective_end']:
        result['coverage'] = scope + '；无精确同日截面，不沿用最后日期或前沿'
        return result
    complete = True
    for stock in universe:
        peer = stock['code']
        if peer == code:
            continue
        try:
            meta = pareto_web._json(context['folder'], f'features/{peer}.json')
            if (not isinstance(meta, dict) or meta.get('code', peer) != peer
                    or meta.get('feature_version', STRATEGY_VERSION) != STRATEGY_VERSION):
                raise ValueError('incompatible peer metadata')
            if meta['action_status'] != 'ok' or meta['rights_issue_present'] is True or meta['feature_error']:
                continue
            if meta['rights_issue_present'] is not False:
                raise ValueError('unknown peer rights')
            with np.load(pareto_web._file(context['folder'], f'features/{peer}.npz'), allow_pickle=False) as data:
                dates, votes = data['dates'], data['votes']
                eligible, gates = data['eligible'], data['gates']
                if (dates.ndim != 1 or dates.dtype.kind != 'U'
                        or any(_day(d) > manifest['effective_end'] for d in dates)
                        or np.any(dates[1:] <= dates[:-1])
                        or votes.shape != (len(dates), 34) or votes.dtype.kind not in 'iuf'
                        or not np.isin(votes, [-1, 0, 1]).all()
                        or any(mask.shape != (len(dates),) or mask.dtype.kind != 'b'
                               for mask in (eligible, gates))):
                    raise ValueError('invalid peer features')
                matches = np.flatnonzero(dates == day)
                if not len(matches):
                    continue  # No signal forward-fill over a suspension.
                i = int(matches[0])
                if i < 250 and eligible[i]:
                    raise ValueError('peer eligibility contradicts warm-up')
                if not (eligible[i] and gates[i]):
                    continue
                other = votes[i]
                result['compared'] += 1
                if np.all(other >= vector) and np.any(other > vector):
                    result['dominators'].append(peer)
        except _DATA_ERRORS:
            complete = False
    if result['dominators']:
        result['status'] = 'dominated'
    elif complete and result['compared']:
        result['status'] = 'front1'
    result['coverage'] = scope + ('' if complete else '；同日截面文件不完整，不确认第一前沿')
    return result


def analyze_stock(code, holding=False):
    if not isinstance(code, str) or not code.startswith(MAINBOARD) or len(code) != 6 or not code.isascii() or not code.isdigit():
        raise ValueError('融合策略仅支持6位沪深主板股票代码')
    if not isinstance(holding, bool):
        raise ValueError('holding必须为布尔值，不能代替真实持仓数量')
    # Refresh is deliberately first: historical evidence cannot short-circuit it.
    fresh = live_data.refresh_stock(code)
    dates, raw, today = _fresh_arrays(fresh)
    context = historical_context(code)
    actions = fresh['actions']
    verified = _verified_actions(actions, code, dates, today)
    adjusted = raw.copy()
    warnings = list(fresh['warnings'])
    events = {}
    if verified:
        try:
            events = aligned_actions(actions['events'], dates)
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                adjusted = causal_adjusted_prices(raw, events)
            if not np.isfinite(adjusted).all():
                raise ValueError('nonfinite adjusted history')
        except _DATA_ERRORS + (FloatingPointError,):
            verified, adjusted, events = False, raw.copy(), {}
    if not verified:
        warnings.append('以下未连续化的原价指标仅作诊断，权息未通过时不用于买卖结论。')
    if dates[-1] < today:
        warnings.append(f'本次在线核验仅取得截至{dates[-1]}的已完成日线；候选/信号仅截至该实际日期，'
                        f'不是核验日{today}行情。未核验交易日历/停牌；15:00前昨日完整日线仍可作截至昨日诊断。')
    if context['manifest']:
        warnings.append(f'风险参数来自冻结报告所选值：年化波动目标{context["risk"].annual_vol_target:.2%}；'
                        f'历史有效截止{context["manifest"]["effective_end"]}不截短本次行情，不套用最后5日退出规则。')
    else:
        warnings.append('无完整历史报告，使用模型默认风险参数：年化波动目标15%，不是训练所选10%。')
    objectives = make_objectives(*adjusted.T)
    stats = risk_statistics(adjusted[:, 3], context['risk'].risk_window, context['risk'].es_confidence)
    i = len(dates)-1
    eligible = bool(objectives['eligible'][i])
    gate = bool(objectives['gate_pass'][i])
    jumps = []
    if verified and len(raw) > 1:
        for j in range(1, len(raw)):
            cash, mult = events.get(j, (0., 1.))
            reference = (raw[j-1, 3]-cash)/mult
            if reference <= 0 or abs(raw[j, 0]/reference-1) > .25:
                jumps.append(str(dates[j]))
    if jumps:
        eligible = False
        warnings.append('存在未解释的超过25%开盘跳空；沿用回测质量隔离，禁止新仓。')
    comparison = compare_same_day(code, str(dates[i]), objectives['votes'][i], context) if verified and eligible and gate else dict(
        status='unknown', compared=0, expected=0, coverage='权息、资格或门禁未通过，未进行前沿比较', dominators=[], date=str(dates[i]))
    vol, es = float(stats['annual_vol'][i]), float(stats['daily_es'][i])
    conditional_cap = risk_target(1, verified and eligible and gate, vol, es, context['risk'])
    # The risk budget may be displayed conditionally; unknown F1 is NOT a buy.
    if not verified:
        action, symbol, css, reason = '暂不下买卖结论', '⚠️', 'warn', '权息核验未通过或存在未建模配股'
    elif not eligible:
        action, symbol, css, reason = '暂不下买卖结论', '⚠️', 'warn', '预热、行情或质量资格未通过'
    elif not gate or comparison['status'] == 'dominated':
        action = '退出信号' if holding else '不买入'
        symbol, css = '❌', 'sell'
        reason = '门禁未通过' if not gate else '被同日其他股票严格支配，不在第一前沿'
    elif comparison['status'] != 'front1':
        action, symbol, css, reason = '暂不下买卖结论', '⚠️', 'warn', '缺少同日可比截面，不能确认F1'
    elif not np.isfinite(vol) or not np.isfinite(es) or conditional_cap <= 0:
        action, symbol, css, reason = '不买入', '⚠️', 'warn', '风险预算不可用或为零'
    else:
        action = '持有候选' if holding else '建仓候选'
        symbol, css, reason = '✅', 'buy', '同日F1、门禁及风险预算通过；不代表已经成交'
    if dates[-1] < today and css in ('buy', 'sell'):
        action = f'截至{dates[-1]}的{action}'
    warnings.append('未提供持股数量、成本、净值高点和冷静期状态，无法计算实际加减仓金额或个人账户回撤退出；候选须核对这些约束。')
    return dict(code=code, name=fresh['name'] or code, holding=bool(holding), fresh=fresh,
                dates=dates, raw=raw, adjusted=adjusted, votes=objectives['votes'],
                eligible=eligible, gate=gate, verified_actions=verified, rsi=float(objectives['rsi'][i]),
                obv_flow=float(objectives['obv_flow'][i]), annual_vol=vol, daily_es=es,
                conditional_cap=conditional_cap, comparison=comparison, action=action,
                symbol=symbol, css=css, reason=reason, history=context, warnings=warnings)
