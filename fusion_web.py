"""HTML adapter for one fresh analysis and its separate frozen account.

render_analysis(code, holding=False, calc_dividend=False) -> str (HTML fragment).
Only analyze_stock performs the targeted refresh/upsert. This module neither
refreshes again nor loads another report, and never substitutes cached analysis
after RefreshError. Dividend display and the holding flag cannot change history.
"""

from datetime import date, datetime
import re

import numpy as np

from fusion_analysis import analyze_stock
from fusion_chart import render_chart
from live_data import RefreshError
from pareto_backtest import MAINBOARD
from pareto_strategy import OBJECTIVE_NAMES
from pareto_web import _decimal, _money, _number, _percent, _same, _table, _text


OBJECTIVE_LABELS = {
    'M_DIF_DEA': '快慢线 DIF / DEA',
    'M_DIF_ZERO': 'DIF 零轴方向',
    'M_DIF_SLOPE_5': 'DIF 五日斜率',
    'M_BAR_TREND_3': 'MACD 柱三日趋势',
    'M_LOW_REPAIR': '底部动能修复',
    'M_TOP_WEAK': '顶部动能衰弱',
    'F_RSI': '相对强弱 RSI',
    'F_K_D': '随机指标 K / D',
    'F_J_EXTREME': '随机指标 J 极值',
    'F_BB_POSITION': '布林带位置',
    'F_WR_EXTREME': '威廉指标极值',
    'P_POSITION_250': '250 日价格区间位置',
    'P_RETURN_250': '250 日涨跌方向',
    'Q_OBV_FLOW': 'OBV 五日净量能流',
    'Q_PRICE_VOLUME_DIVERGENCE': '量价背离',
    'Q_INSTITUTION_PROXY': '波动率与价格行为代理（非真实机构资金）',
    'A_DMI': '趋向指标', 'A_CCI': '顺势指标',
    'A_BIAS(6)': '六日乖离率', 'A_BIAS(12)': '十二日乖离率',
    'A_EXPMA': '指数均线', 'A_BBI': '多空均线',
    'A_TRIX': '三重指数平滑', 'A_VR': '成交量变异率',
    'A_BR': '意愿指标', 'A_AR': '人气指标', 'A_CR': '中间意愿指标',
    'A_DMA': '平均线差', 'A_DPO': '去趋势价格振荡', 'A_MTM': '动量指标',
    'A_SKDJ': '慢速随机指标', 'A_LWR': '慢速威廉指标',
    'A_ENE': '均线轨道位置', 'A_LON': '长线通道位置',
}
_GROUPS = (('M', 'MACD 动能 · M6'), ('F', '多因子状态 · F5'),
           ('P', '价格位置与趋势 · P2'), ('Q', '量价代理 · Q3'),
           ('A', '辅助指标 · Aux18'))
_FILLS = dict(build='建仓', add='增持', reduce='减持', exit='退出')
_DATA_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError,
                AttributeError, OverflowError, RuntimeError)
_LEGEND = '✅条件通过/❌条件未通过/⚠️数据不足，非盈利概率；各坐标↑+1/→0/↓−1'


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
        raise ValueError('invalid day')
    return date.fromisoformat(value)


def _diagnostic(value, percent=False):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return '—（数据不足）'
    return _percent(value) if percent else _decimal(value)


def _scroll(headers, rows, css='trades'):
    return ('<div class="table-scroll" tabindex="0" role="region" aria-label="结果表格，可横向滚动">'
            + _table(headers, rows, css) + '</div>')


def _failure(message, refresh_id=None):
    # Never echo exception args/evidence_dir: they can contain absolute paths or
    # remote bodies. Only the opaque refresh ID is useful to a browser client.
    identity = ''
    if isinstance(refresh_id, str) and not re.match(r'^(?:[/\\]|[A-Za-z]:)', refresh_id):
        identity = '<p class="hint">刷新 ID：' + _text(refresh_id) + '</p>'
    return ('<div class="result fusion-report"><div class="conclusion warn">⚠️ '
            + _text(message) + '</div>' + identity
            + '<p class="hint">不展示旧缓存结论或图表，不以历史回测代替本次更新。</p></div>')


def _validate_analysis(a, code):
    """Fail closed on mixed dates/vectors or an inconsistent positive action."""
    if a['code'] != code or set(OBJECTIVE_LABELS) != set(OBJECTIVE_NAMES):
        raise ValueError('analysis identity mismatch')
    days = np.asarray(a['dates'])
    if days.ndim != 1 or not len(days) or days.dtype.kind != 'U':
        raise ValueError('missing daily history')
    for day in days:
        _day(day)
    if np.any(days[1:] <= days[:-1]):
        raise ValueError('unordered daily history')
    fresh = a['fresh']
    if fresh['latest_date'] != days[-1] or fresh.get('date', days[-1]) != days[-1]:
        raise ValueError('mixed analysis dates')
    clock = datetime.fromisoformat(fresh['checked_at'])
    if clock.utcoffset() is None or _day(days[-1]) > clock.date():
        raise ValueError('invalid refresh clock')
    for key, columns in (('raw', 5), ('adjusted', 5), ('votes', 34)):
        values = np.asarray(a[key])
        if (values.shape != (len(days), columns) or values.dtype.kind not in 'iuf'
                or not np.isfinite(values).all()):
            raise ValueError('invalid analysis array')
        if key == 'votes':
            if not np.isin(values, (-1, 0, 1)).all():
                raise ValueError('invalid votes')
        elif (np.any(values[:, :4] <= 0) or np.any(values[:, 4] < 0)
              or np.any(values[:, 2] > np.minimum(values[:, 0], values[:, 3]))
              or np.any(values[:, 1] < np.maximum(values[:, 0], values[:, 3]))):
            raise ValueError('invalid OHLCV')
    if a['css'] not in ('buy', 'sell', 'warn'):
        raise ValueError('invalid conclusion class')
    actions = fresh['actions']
    verified = (actions['status'] == 'ok' and actions['rights_issue_present'] is False
                and not actions['metadata'].get('live_recommendation_blocked', True))
    if not verified and a['css'] != 'warn':
        raise ValueError('unverified action conclusion')
    if a['css'] == 'buy' and not (
            verified and a['verified_actions'] and a['eligible'] and a['gate']
            and a['comparison']['status'] == 'front1'
            and np.isfinite(a['annual_vol']) and np.isfinite(a['daily_es'])
            and _number(a['conditional_cap']) > 0):
        raise ValueError('unconfirmed F1 cannot be a buy')


def _current(a, holding):
    fresh, comparison = a['fresh'], a['comparison']
    status = {'front1': 'F1 条件通过（仅此同日截面）',
              'dominated': '不在 F1：存在同日严格支配者',
              'unknown': 'F1 未知：不能确认，不据此买入'}.get(comparison['status'], 'F1 未知')
    rows = [
        ('联网核验时间 checked_at（非行情日期）', _text(fresh['checked_at'])),
        ('来源最新完整日线日期', _text(fresh['latest_date'])),
        ('行情提供方 provider', _text(fresh['provider'])),
        ('本次分析行情范围', _text(a['dates'][0]) + ' ～ ' + _text(a['dates'][-1])),
        ('最新不复权收盘价', _money(np.asarray(a['raw'])[-1, 3])),
        ('权息核验', '通过' if a['verified_actions'] else '未通过：以下原价指标仅供诊断'),
        ('资格 eligible / 门禁 gate', ('通过' if a['eligible'] else '未通过')
         + ' / ' + ('通过' if a['gate'] else '未通过')),
        ('同日比较 / F1', status + '；比较日 ' + _text(comparison.get('date'))),
        ('截面覆盖', _text(comparison['coverage']) + '；已比较有效同日股票 '
         + _text(comparison['compared']) + '；冻结池其他股票 ' + _text(comparison['expected'])),
        ('条件风险上限（非执行目标）', _diagnostic(a['conditional_cap'], True)),
        ('当前持仓选项', ('已持仓' if holding else '未持仓')
         + '；仅影响当前动作，不改变历史账户'),
    ]
    return ('<div class="current-analysis"><h3>当前分析</h3>'
            + _scroll(('项目', '本次联网分析'), rows, 'overview')
            + '<p class="hint">核验日不等于行情日；只取得来源最新已完成日线，不称盘中实时价。'
            '条件风险上限须先满足同日 F1 等约束，F1 未知时不能作为买入依据，'
            '不是执行目标或实际仓位。历史账户仓位及最后五日退出规则不套用于当前持仓选项。</p></div>')


def _indicators(a):
    n = len(a['dates'])
    parts = ['<section class="strategy indicators"><h3>34 个独立多维指标</h3>',
             '<p class="hint">逐坐标展示方向强弱，不求和；偏强/偏弱不是盈利概率，0 不是中性推荐。</p>']
    if n < 251:
        parts.append(f'<p class="method-note">原子预热未完成：仅 {n} 根日线，完整资格至少需要 251 根；'
                     '当前所有行仅供诊断，预热占位 0 不能当作中性推荐。辅助原子前 59 根为占位 0。</p>')
    if not a['eligible']:
        parts.append('<p class="method-note">资格未通过：预热、成交量或质量隔离可能不满足；不能据诊断投票建仓。</p>')
    parts.append('<div class="formula-grid">')
    votes = np.asarray(a['votes'])[-1]
    for prefix, title in _GROUPS:
        rows = []
        for name, vote in zip(OBJECTIVE_NAMES, votes):
            if not name.startswith(prefix + '_'):
                continue
            label, color, css = {1: ('↑ +1 偏强', '#2e7d32', 'up'),
                                 0: ('→ 0 无方向', '#64748b', 'flat'),
                                 -1: ('↓ −1 偏弱', '#c62828', 'down')}[int(vote)]
            unready = ((prefix == 'A' and n < 60) or (name == 'P_POSITION_250' and n < 250)
                       or (name == 'P_RETURN_250' and n < 251))
            direction = f'<span class="vote vote-{css}" style="color:{color};font-weight:700">{label}</span>'
            if unready:
                direction += '<span class="hint">（预热未完成，仅诊断）</span>'
            rows.append((_text(name), _text(OBJECTIVE_LABELS[name]), direction))
        parts.append(f'<div class="formula-section" data-group="{prefix}"><h4>{title}</h4>'
                     + _table(('独立原子', '中文含义', '方向'), rows, 'overview') + '</div>')
    parts.append('</div><h3>诊断统计（非评分）</h3>' + _scroll(('指标', '当前值'), [
        ('RSI 相对强弱', _diagnostic(a['rsi'])),
        ('OBV 五日净量能流（非真实资金流）', _diagnostic(a['obv_flow'], True)),
        ('年化波动率 σ', _diagnostic(a['annual_vol'], True)),
        ('日历史 ES', _diagnostic(a['daily_es'], True)),
    ], 'overview') + '</section>')
    return ''.join(parts)


def _annual(account):
    rows = []
    previous = _number(account['initial_capital'])
    for year, record in sorted(account.get('years', {}).items()):
        nav = _number(record['nav'])
        rows.append((_text(year), _text(record['date']), _money(nav), _money(nav - previous),
                     _percent(nav / previous - 1) if previous > 0 else '—',
                     _money(record['cash']), _decimal(record['shares'])))
        previous = nav
    return ('<details class="historical-years"><summary>历史年度收益（展开）</summary>'
            '<p class="hint">本期相对上一条年末权益，首期相对初始本金；含已入账分红和标记残值。'
            '年度缺口不插补，不冒充连续单年收益；首末年可能不足一年，分段不重置账户。</p>'
            + (_scroll(('年份', '记录日期', '权益', '本期盈亏', '本期收益率', '现金', '股份'), rows)
               if rows else '<p class="hint">暂无年度账户记录，不推算年度收益。</p>') + '</details>')


def _history(context, code):
    """Use only this stock's account; never replace it with population totals."""
    manifest = context.get('manifest') or {}
    date_rows = [('回测起始（冻结清单）', _text(manifest.get('start'))),
                 ('回测请求截止', _text(manifest.get('requested_end'))),
                 ('回测有效截止', _text(manifest.get('effective_end')))]
    opening = '<div class="historical-summary"><h3>历史回测摘要</h3>'
    missing = '<p class="hint">暂无该股可用的完整历史账户；不推算收益，不用全市场汇总代替。</p>'
    account = context.get('account')
    if not account:
        return (opening + _scroll(('历史独立账户', '金额 / 日期'), date_rows, 'overview')
                + missing + '<p class="hint">' + _text(context.get('error')) + '</p></div>', '', None)
    try:
        end = manifest['effective_end']
        _day(end)
        if account['code'] != code or account['end'] != end:
            raise ValueError('mixed account')
        initial = _number(account['initial_capital'])
        _same(account['marked_nav'], _number(account['final_cash']) + _number(account['residual_value']))
        _same(account['pnl'], _number(account['marked_nav']) - initial)
        rows = date_rows + [(label, _money(account[key])) for key, label in (
            ('initial_capital', '实际初始本金'), ('final_cash', '期末现金'),
            ('residual_value', '残余股份标记残值（非现金）'), ('marked_nav', '期末权益＝现金＋残值'),
            ('pnl', '总盈亏＝权益−本金'), ('fees', '已扣显式费用'),
            ('cash_dividends', '已入账现金分红（已含在总收益）'))]
        rows += [('本期总收益率（非年化）', _percent(account['pnl'] / initial) if initial > 0 else '—'),
                 ('期末持股 / 估值日期', _decimal(account['shares']) + ' 股 / ' + _text(account['last_date'])),
                 ('建仓 / 增持 / 减持 / 退出次数', ' / '.join(_text(account[k + '_count']) for k in _FILLS)),
                 ('未成交尝试 / 权息事件数', _text(account['unfilled_count']) + ' / ' + _text(account['action_count']))]
        ledger = context['ledger']
        if not isinstance(ledger, list):
            raise ValueError('invalid ledger')
        fills, last_day = [], ''
        for entry in ledger:
            day = entry['day']
            _day(day)
            if day < last_day or day > end or not isinstance(entry['event'], str):
                raise ValueError('invalid ledger date or event')
            last_day = day
            if entry['event'] in _FILLS:
                if _number(entry['quantity']) <= 0 or _number(entry['price']) <= 0:
                    raise ValueError('invalid fill')
                if entry.get('signal_day') is not None and _day(entry['signal_day']) >= _day(day):
                    raise ValueError('noncausal fill')
                fills.append(entry)
        fill_rows = [(_text(e['day']), _text(e.get('signal_day')), _text(_FILLS[e['event']]),
                      _decimal(e['quantity']), _decimal(e['price']), _money(e['fee']),
                      _money(e['cash']), _decimal(e['shares']), _text(e.get('reason')))
                     for e in fills[-20:]]
        detail = ('<section class="historical-detail">' + _annual(account)
                  + f'<h3>最近实际成交账本：共 {len(fills)} 次，展示最近 {min(20, len(fills))} 次</h3>'
                  + '<p class="hint">仅 build/add/reduce/exit 的真实 filled 账本事件；非实盘成交。'
                  '未成交和权息事件不冒充成交，不配对交易、不推算胜率；信号日与执行日分开。</p>'
                  + (_scroll(('执行日', '信号日', '动作', '数量（股）', '成交价（元）',
                              '费用', '成交后现金', '成交后股份', '原因'), fill_rows)
                     if fill_rows else '<p class="hint">历史账户无实际成交，全部本金仍保留在账户统计中。</p>')
                  + '</section>')
        top = (opening + _scroll(('历史独立账户', '金额 / 日期'), rows, 'overview')
               + '<p class="hint">冻结回测独立于本次行情日期；费用已扣、现金分红已入账，不重复加减。'
               '持仓及分红开关均不改变总收益；残值不等于已兑现现金，显式费用不含已计入成交价的滑点。</p></div>')
        return top, detail, dict(end=end, ledger=ledger)
    except _DATA_ERRORS:
        return (opening + _scroll(('历史独立账户', '金额 / 日期'), date_rows, 'overview')
                + '<p class="method-note">历史账户或账本不完整、资金不对齐，暂不展示历史收益及成交标记。</p></div>', '', None)


def _dividends(a):
    actions, account = a['fresh']['actions'], (a.get('history') or {}).get('account')
    parts = ['<section class="dividend-history"><h3>最新取得的分红历史</h3>',
             '<p class="hint">取得时间：' + _text(actions.get('retrieved_at'))
             + '；权息状态：' + _text(actions.get('status')) + '</p>']
    try:
        asof = datetime.fromisoformat(a['fresh']['checked_at']).date()
        # Twelve calendar months, including leap-day handling, not a fixed 365d.
        lower = asof.replace(year=asof.year - 1, day=28 if asof.month == 2 and asof.day == 29 else asof.day)
        rows, cash12 = [], 0.
        for event in sorted(actions['events'], key=lambda e: e['ex_date']):
            ex = _day(event['ex_date'])
            cash, mult = _number(event['cash_per_share']), _number(event['share_multiplier'])
            if cash < 0 or mult <= 0 or ex > asof:
                raise ValueError('invalid action event')
            if lower < ex <= asof:
                cash12 += cash
            rows.append((_text(event['ex_date']), _decimal(cash), _decimal(mult)))
        parts.append(_scroll(('除权除息日 ex_date', '税前现金（元/权息前股）', '送转后/前股份倍数'), rows)
                     if rows else '<p class="hint">本次返回的现金/送转事件列表为空。</p>')
        if actions['status'] == 'ok':
            parts.append('<p class="hint">近12月已取得每股现金简单合计：'
                         + _decimal(cash12) + ' 元/各次权息前股；窗口 (' + lower.isoformat()
                         + ', ' + asof.isoformat() + ']（按联网核验日，不按回测截止）。</p>')
        else:
            parts.append('<p class="method-note">权息核验未通过：列表仅供诊断，近12月现金合计不可用；空列表不等于零分红。</p>')
        if account is not None:
            parts.append('<p class="hint">历史账户已入账现金分红：' + _money(account['cash_dividends'])
                         + '，已包含在历史总收益中，不再添加。</p>')
    except _DATA_ERRORS:
        parts.append('<p class="method-note">分红事件或账户字段不完整，不能计算近12月每股现金合计。</p>')
    parts.append('<p class="hint">仅列本次取得的 actions.events，不另行抓取、不以旧缓存补表。'
                 '送转前后每股基数未归一化，简单合计不是同一初始股的持有收益，不计算股息率，'
                 '也不乘送转倍数后叠加到账户收益。除权日不等于到账日；红利税、送转股可用日及供应商历史完整性未核验。'
                 '此附表可独立于持仓选项展示；关闭附表不会从账本剔除分红。</p></section>')
    return ''.join(parts)


def _notes(a):
    context = a.get('history') or {}
    manifest, summary = context.get('manifest') or {}, context.get('summary') or {}
    risk = context.get('risk')
    parameters = ''
    if risk is not None:
        parameters = ('<p class="hint">' + ('本轮冻结所选风险参数' if context.get('manifest') else '无历史报告，模型默认风险参数')
                      + '：年化波动目标 ' + _percent(risk.annual_vol_target)
                      + '；日 ES 预算 ' + _percent(risk.es_budget) + '；最大上限 ' + _percent(risk.max_weight)
                      + '；风险窗口 ' + _text(risk.risk_window) + ' 个收益日，ES 置信度 '
                      + _percent(risk.es_confidence) + '（尾部统计参数，非盈利概率）。</p>')
    notes = list(dict.fromkeys(list(manifest.get('limitations', [])) + list(summary.get('limitations', []))))
    return ('<details class="strategy methodology"><summary>方法与限制（展开）</summary>' + parameters
            + '<p class="hint">34 个三值坐标分别比较，不汇总成置信分；独立不等于统计独立，三值化丢失幅度。'
            '同日有效、预热完成且通过门禁的截面才可比较；所有坐标不差且至少一项更优才构成严格支配，'
            '相同向量不互相支配。F1 不提供唯一最优解，也不意味着满仓或保证盈利。</p>'
            '<p class="hint">条件风险上限＝min(最大上限, 波动目标/年化波动率, ES预算/日ES)，风险分母下限 1e−8；'
            '门禁为 RSI≤92 且 OBV五日净流≥−60%。未知 F1 不按单股自封第一前沿。'
            '未提供个人持股数量、成本、净值高点和冷静期，不能计算实际加减仓金额或个人回撤退出。</p>'
            '<p class="hint">历史账本：收盘信号→下一市场日开盘尝试成交；T+1、整手、费用和成交限制仍适用。'
            '现金分红近似除权日入账，送转股近似当日可用，未计红利税及零碎股近似；配股未建模。'
            '最后5个市场日禁买并尝试退出，仅属于历史回放截止规则。日线涨跌停代理、无参与率限制、'
            '陈旧残值及事后权息质量隔离均有局限；回撤退出不是保本或最大亏损保证。</p>'
            + '<ul>' + ''.join('<li>' + _text(note) + '</li>' for note in notes) + '</ul>'
            + ('<p class="hint">本轮历史研究结论：' + _text(summary['verdict']) + '</p>' if summary.get('verdict') else '')
            + '</details>')


def render_analysis(code, holding=False, calc_dividend=False) -> str:
    """Return a result fragment; invalid codes never reach network/analysis.

    The first conclusion comes from the analysis contract, never vote totals.
    On success the final content element is the PNG image. Chart failure leaves
    the diagnostic HTML available, without a stale image or a second refresh.
    """
    if (not isinstance(code, str) or re.fullmatch(r'[0-9]{6}', code) is None
            or not code.startswith(MAINBOARD)):
        return _failure('请输入6位沪深主板股票代码')
    try:
        a = analyze_stock(code, holding=holding)
    except RefreshError as error:
        return _failure('更新失败，暂不下结论', error.refresh_id)
    except _DATA_ERRORS:
        return _failure('分析失败，暂不下结论')
    try:
        _validate_analysis(a, code)
        top_history, detail_history, chart_history = _history(a.get('history') or {}, code)
        warnings = list(dict.fromkeys(list(a.get('warnings', []))
                        + list(a['fresh'].get('warnings', []))
                        + list(a['fresh']['actions']['metadata'].get('notes', []))))
        parts = ['<div class="result fusion-report">',
                 '<div class="conclusion ' + a['css'] + '">' + _text(a['symbol']) + ' '
                 + _text(a['action']) + ' · 数据日期 ' + _text(a['fresh']['latest_date']) + '</div>',
                 '<h2>' + _text(a['name']) + ' (' + _text(code) + ') — 融合策略</h2>',
                 '<p class="hint">' + _text(a['reason']) + '</p>',
                 '<p class="hint symbol-legend">' + _LEGEND + '</p>',
                 '<div class="top-row">' + _current(a, holding) + top_history + '</div>',
                 '<div class="analysis-warnings">'
                 + ''.join('<p class="method-note">' + _text(w) + '</p>' for w in warnings) + '</div>',
                 _indicators(a), detail_history]
        if calc_dividend:
            parts.append(_dividends(a))
        parts.append(_notes(a))
        parts.append('<p class="hint">下图价格为不复权价格；'
                     + ('MACD 使用权息连续化价格。' if a['verified_actions'] else
                        '权息未通过，MACD 实际输入为未连续化原价，仅作诊断。')
                     + ('历史成交标记仅来自冻结账本，有效截止 ' + _text(chart_history['end'])
                        + '；截止之后只展示行情及信号，无后续成交标记。' if chart_history else
                        '无可用历史成交账本，仅展示行情和信号，不推断成交或收益。') + '</p>')
    except _DATA_ERRORS:
        return _failure('分析数据不完整，暂不下结论')
    try:
        png = render_chart(a['dates'], a['raw'], a['adjusted'], a['votes'], history=chart_history,
                   action_continuous=a['verified_actions'])
        if not isinstance(png, str) or not png:
            raise ValueError('missing PNG')
        parts.append('<img src="data:image/png;base64,' + _text(png)
                     + '" alt="融合策略诊断图：价格、MACD、34维独立信号及可用历史成交">')
    except _DATA_ERRORS:
        parts.append('<p class="hint chart-error">图表暂不可用；以上诊断保留，不展示旧图、不重新刷新。</p>')
    return ''.join(parts) + '</div>'