"""Read-only HTML views of a completed daily Pareto run; no DB/network fallback.

summary.json is the runner's completion marker. Paths come only from service
configuration, never from HTTP parameters. No runner, refresh or scoring entry
point is invoked here; fronts and ledger targets remain frozen run artifacts.
"""

from datetime import date
from html import escape
import json
import math
import os
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np

from capital_account import AccountConfig
from pareto_strategy import OBJECTIVE_NAMES, STRATEGY_VERSION, RiskConfig, risk_target

ROOT = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get('PARETO_RUN_DIR') or ROOT / 'output' / 'pareto_mainboard_20260910')
MISSING_RUN = ('尚无完成的多维回测，请先运行 pareto_backtest.py --output '
               'output/pareto_mainboard_20260910，并等待 summary.json 完成。'
               '自定义运行目录仅由服务端 PARETO_RUN_DIR 配置。')


class ReportError(ValueError):
    """An incomplete/incompatible artifact is not an invitation to recompute."""


def _text(value):
    return escape('—' if value is None else str(value), quote=True)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ReportError('本地回测数值格式不完整，请重新生成完整产物。')
    number = float(value)
    if not math.isfinite(number):
        raise ReportError('本地回测资金记录含非有限值，请核验产物。')
    return number


def _money(value):
    return f'{_number(value):,.2f} 元'


def _decimal(value):
    return '—' if value is None else f'{_number(value):,.4f}'


def _percent(value):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return '—'
    return f'{_number(value) * 100:.2f}%'


def _same(left, right):
    if not math.isclose(_number(left), _number(right), rel_tol=1e-9, abs_tol=.02):
        raise ReportError('本地回测资金不对齐，请核验现金、残值、权益和本金；未回落旧策略。')


def _iso(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ReportError('本地回测日期格式不正确。')
    return value


def _file(folder, relative):
    root = Path(folder).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ReportError('产物路径超出服务配置目录。')
    return path


def _json(folder, relative):
    with _file(folder, relative).open(encoding='utf-8') as stream:
        return json.load(stream)


def _load_run():
    folder = Path(RUN_DIR)
    if not _file(folder, 'summary.json').is_file():
        raise ReportError(MISSING_RUN)
    summary = _json(folder, 'summary.json')
    manifest = _json(folder, 'manifest.json')
    if any(item['version'] != STRATEGY_VERSION for item in (manifest, summary)):
        raise ReportError('多维回测版本不匹配，请使用当前策略重新生成完整产物。')
    if tuple(manifest['objective_names']) != tuple(OBJECTIVE_NAMES):
        raise ReportError('多维回测版本不匹配：34 维原子名称或顺序不同。')
    for key in ('start', 'requested_end', 'effective_end'):
        _iso(manifest[key])
        if summary[key] != manifest[key]:
            raise ReportError('回测汇总与清单日期不一致，请等待同一轮完整产物。')
    calendar = manifest['calendar']
    if (not calendar or calendar != sorted(set(calendar))
            or any(not manifest['start'] <= _iso(day) <= manifest['effective_end'] for day in calendar)
            or manifest['effective_end'] > manifest['requested_end']):
        raise ReportError('本地回测市场日历不完整或不一致。')
    codes = [row['code'] for row in manifest['universe']]
    if (not codes or len(set(codes)) != len(codes)
            or any(not isinstance(code, str) or not re.fullmatch(r'[0-9]{6}', code) for code in codes)):
        raise ReportError('本地回测股票清单不完整。')
    risk = RiskConfig(**summary['selected_risk'])
    account_config = AccountConfig(**manifest['account_defaults'])
    RiskConfig(**manifest['risk_defaults'])
    totals = summary['totals']
    if totals['account_count'] != len(codes):
        raise ReportError('回测账户数量与股票清单不一致。')
    _same(totals['initial_capital'], len(codes) * account_config.initial_capital)
    _same(totals['final_equity'], _number(totals['final_cash']) + _number(totals['residual_marked_value']))
    _same(totals['pnl'], _number(totals['final_equity']) - _number(totals['initial_capital']))
    _same(totals['cash_minus_all_principal'], _number(totals['final_cash']) - _number(totals['initial_capital']))
    if _iso(manifest['train_end']) >= _iso(manifest['validation_end']):
        raise ReportError('训练与验证区间不一致。')
    trials = summary['trials']
    if not trials or [t['vol_target'] for t in trials] != manifest['candidate_vol_targets']:
        raise ReportError('风险参数试验与清单不一致。')
    for trial in trials:
        if trial['scope_end'] != manifest['effective_end']:
            raise ReportError('风险试验须连续运行至同一有效截止日，不能在验证末日提前清仓。')
        _same(trial['validation_end_nav'], _number(totals['initial_capital'])
              + _number(trial['train_pnl']) + _number(trial['validation_pnl']))
    selected = max(trials, key=lambda t: (_number(t['train_pnl']), -_number(t['vol_target'])))
    if risk.annual_vol_target != selected['vol_target']:
        raise ReportError('风险参数与仅训练期选优结果不一致。')
    _same(summary['holdout_2024_onward_pnl'], _number(totals['final_equity'])
          - _number(selected['validation_end_nav']))
    previous = _number(totals['initial_capital'])
    last_date = ''
    if not summary['annual_results']:
        raise ReportError('年度资金记录不完整。')
    for record in summary['annual_results']:
        day = _iso(record['date'])
        if day not in calendar or day <= last_date or str(record['year']) != day[:4]:
            raise ReportError('年度资金日期与日历不一致。')
        equity = _number(record['final_equity'])
        _same(record['pnl'], equity - previous)
        if previous:
            _same(record['return_pct'], (equity / previous - 1) * 100)
        previous, last_date = equity, day
    _same(previous, totals['final_equity'])
    return folder, manifest, summary, risk, account_config


def _load_account(folder, code, config, end):
    account = _json(folder, f'accounts/{code}.json')
    if account['code'] != code:
        raise ReportError('账户代码与产物文件不一致。')
    if account['end'] != end:
        raise ReportError('账户回放截止与完成运行不一致。')
    _same(account['initial_capital'], config.initial_capital)
    _same(account['cash'], account['final_cash'])
    _same(account['residual_value'], _number(account['shares']) * _number(account['last_price']))
    _same(account['marked_nav'], _number(account['final_cash']) + _number(account['residual_value']))
    _same(account['pnl'], _number(account['marked_nav']) - _number(account['initial_capital']))
    if account['liquidation_complete'] != (account['shares'] == 0):
        raise ReportError('账户清仓状态与残余股份不一致。')
    return account


def _table(headers, rows, css='trades'):
    # Cell strings are already escaped/formatted by the caller; headers are text.
    return (f'<table class="{css}"><thead><tr>'
            + ''.join(f'<th>{_text(header)}</th>' for header in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows)
            + '</tbody></table>')


def render_methodology(risk=None, account=None):
    """Shared homepage/report documentation, with model or frozen run defaults."""
    parameter_label = '模型默认参数（本次实际参数以分析结果为准）' if risk is None else '本轮冻结参数'
    risk = risk or RiskConfig()
    account = account or AccountConfig()
    return f'''
    <p class="method-note">每日 Pareto34 原子：34 个独立维度各投 +1 / 0 / −1，不先合并共识、不加权、不按收益挑 Top-N。
    同一市场日符合资格及门禁的股票比较非支配关系，全部第一前沿 F1 保留；F1 不等于满仓。
    支配要求所有维度不差且至少一维更优；相同向量同属一层，不因展示数量裁掉前沿成员。
    独立坐标不等于统计独立；三值化丢失幅度，不是保留全部信息，第一前沿也不提供唯一最优解或盈利保证。</p>
    <p class="hint">历史回测每只股票独立初始账户 {_money(account.initial_capital)}（默认 100 万元），互不借资、不跨股归一化。
    缺历史、停牌、失败账户和闲置现金仍计入全部本金；闲置现金零利息，不保本。</p>
    <div class="formula-section"><h4>动态风险预算（不是 F1 = 100%）</h4>
    <div class="formula-math">wcap = min(1, σTarget / σ, ESbudget / ES95)；若配置 max_weight 更低，再取其上限。<br>
    min [0.5 λ σ² (w − wcap)² + c± |w − wprev|]，硬约束 0 ≤ w ≤ wcap；目标为 0 强制 exit，不因费用而继续持有。</div>
    <p class="hint">{parameter_label}：σTarget={_percent(risk.annual_vol_target)}；ESbudget={_percent(risk.es_budget)}；
    λ={_decimal(risk.tracking_penalty)}；max_weight={_percent(risk.max_weight)}；风险窗口 {_text(risk.risk_window)} 个收益日，
    历史 ES 置信度 {_percent(risk.es_confidence)}（默认 ES95）。波动率为简单收益样本标准差 × √252，
    ES 为窗口最差尾部的非负日亏损均值；缺失风险、非 F1、资格或门禁未通过时目标为 0。
    风险分母下限 1e−8；成本优化的方差下限 1e−8。c± 为买卖两侧比例费用代理，不是给 34 维加权。
    Moreira–Muir 波动管理与 cvxportfolio 风险/成本优化仅提供概念参考，不为本策略公式、参数或盈利背书。</p></div>
    <div class="formula-section"><h4>历史回测账本执行与假设（非当前个人持仓）</h4>
    <p class="hint">收盘信号 → 下一市场交易日开盘执行（signal close → next open），T+1；停牌后过期买单不执行，退出请求可续行。
    买入及非零目标调减按 100 股整手，目标 0 尝试退出全部可卖股份，实际资金与最低费用在账本结算。<br>
    账户净值回撤 {_percent(account.max_drawdown)}（默认 15%）触发 exit + {_text(account.cooldown_sessions)} 个市场日 cooldown（默认 20 日）；
    最后 5 个市场日 deadline 窗口禁止买入并尝试退出，公共截止日可另作收盘清仓尝试。这是预先已知的历史截止规则，不是同日预测成交，不套用于当前单股分析。<br>
    双边佣金 {_percent(account.commission_rate)}、每笔最低 {_money(account.minimum_commission)}；滑点 {_decimal(account.slippage_bps)} bp（默认 5bp）。
    卖出印花税在 2023-08-28 前 0.1%、之后 0.05%；双边过户费在 2022-04-29 前统一假设 0.002%、之后 0.001%，不是完整历史费率复原。<br>
    不复权价成交，权息连续化价格仅供信号；现金分红按除权日到账近似（cash dividend on ex-date），送转股份也近似当日可用，未计红利税，配股未建模。
    权息抓取失败或未核验（action_status ≠ ok）、存在配股历史的两类账户保守隔离：全期禁止买入，eligible=false，仍保留并统计全部本金；不是剔除亏损样本，也不是保本承诺。
    使用开盘不利跳空 ≥4.8% 的保守涨跌停代理，日线不能证明成交，无成交量参与上限；未解释的 &gt;25% 跳空隔离后续信号。
    缺失或无效行情仅延续最后有效估值，不重置本金、不前填交易信号；零碎股份出售为近似，未按真实现金替代款结算。
    无法清仓的股份按最后可用价标记残值，不能把残值当现金或宣称完整总收益已验证。</p></div>'''


def _header(manifest, title):
    return (f'<h2>{_text(title)}</h2><p class="method-note">完成运行的本地历史快照，不联网、不刷新、不重算前沿。'
            f'回测起始 {_text(manifest["start"])}；请求截止 {_text(manifest["requested_end"])}；'
            f'有效截止日期 {_text(manifest["effective_end"])}。旧缓存并非实时推荐，不能当作今日买卖建议。</p>'
            f'<p class="hint">策略版本：{_text(manifest["version"])}</p>')


def _limitations(manifest, summary):
    items = list(dict.fromkeys(manifest['limitations'] + summary['limitations']))
    return (f'<h3>研究边界</h3><p class="hint">训练期截至 {_text(manifest["train_end"])}，'
            f'之后至 {_text(manifest["validation_end"])} 为验证期；'
            '仅训练期货币盈亏选择风险参数，同分取较低波动目标，验证期不参与选优，2024 年起为设计上的留出分段。'
            '本轮已经多次查看后段结果，不是完全独立盲测；不能看过后调参再冒充新留出验证。'
            '各 trial 与最终账户连续运行完整区间、共用同一有效截止日；训练/验证切点只读净值，不额外清仓或重置账户。'
            '所选 trial 验证末权益须等于最终留出期起点权益。'
            '当前缓存股票池存在幸存者偏差；数据覆盖不齐，历史表现不保证未来收益。</p><ul>'
            + ''.join(f'<li>{_text(item)}</li>' for item in items)
            + f'</ul><p class="hint">运行结论：{_text(summary["verdict"])}</p>')


def _action_blocked(account):
    return account['action_status'] != 'ok' or account['rights_issue_present']


def _quality(account):
    return '；'.join((
        '权息状态 ' + _text(account['action_status']),
        '存在配股（未建模）' if account['rights_issue_present'] else '未标记配股（不代表已完整核验）',
        '保守隔离：全期禁止买入，保留本金' if _action_blocked(account) else '权息账户隔离未触发（仍须通过资格与门禁）',
        '未解释跳空 ' + _text(account['unexplained_gap_count']),
        '特征错误 ' + _text(account['feature_error']),
        '晚起始 ' + ('是' if account['late_start'] else '否'),
        '陈旧报价 ' + ('是' if account['stale'] else '否'),
    ))


def _snapshot(folder, manifest, account, risk):
    code = account['code']
    with np.load(_file(folder, f'features/{code}.npz'), allow_pickle=False) as data:
        dates, votes = data['dates'], data['votes']
        vol, es = data['annual_vol'], data['daily_es']
        eligible, gates, raw = data['eligible'], data['gates'], data['raw']
    n = len(dates)
    if (dates.ndim != 1 or dates.dtype.kind != 'U' or votes.shape != (n, len(OBJECTIVE_NAMES))
            or votes.dtype != np.dtype('int8') or not np.isin(votes, [-1, 0, 1]).all()
            or any(a.shape != (n,) for a in (vol, es, eligible, gates))
            or eligible.dtype.kind != 'b' or gates.dtype.kind != 'b'
            or vol.dtype.kind not in 'fiu' or es.dtype.kind not in 'fiu'
            or raw.shape != (n, 5) or raw.dtype.kind not in 'fiu'
            or (n > 1 and not np.all(dates[1:] > dates[:-1]))):
        raise ReportError('34 维特征形状、投票或日期顺序不匹配，请重新生成完整产物。')
    fronts = np.load(_file(folder, 'fronts.npy'), mmap_mode='r', allow_pickle=False)
    try:
        if fronts.shape != (len(manifest['calendar']), len(manifest['universe'])) or fronts.dtype.kind not in 'iu':
            raise ReportError('前沿矩阵与全局日历、股票顺序不匹配。')
        day = account['last_date']
        matches = np.flatnonzero(dates == day) if day else []
        if not len(matches) or day not in manifest['calendar']:
            return '<h3>34 维同日快照</h3><p class="method-note">账户末日、特征与市场日历无精确同日匹配，无法展示 F1 状态或风险上限；不前填旧信号。</p>'
        i = int(matches[0])
        ci = manifest['calendar'].index(day)
        col = [row['code'] for row in manifest['universe']].index(code)
        layer = int(fronts[ci, col])
    finally:
        # Release the read-only mapping on Windows too; no cross-request cache.
        if isinstance(fronts, np.memmap):
            fronts._mmap.close()
    action_blocked = _action_blocked(account)
    valid_bar = bool(np.isfinite(raw[i]).all() and (raw[i] > 0).all())
    effective_eligible = bool(eligible[i]) and not action_blocked and valid_bar
    cap = risk_target(layer, bool(effective_eligible and gates[i]), float(vol[i]), float(es[i]), risk)
    status = 'F1（第一前沿）' if layer == 1 else '不在 F1（含无资格情况，不推断其他层次）'
    deadline = ci >= max(0, len(manifest['calendar']) - 5)
    details = _table(('快照项', '结果'), [
        ('精确匹配日期', _text(day)), ('同日非支配状态（冻结前沿记录）', status),
        ('特征 eligible 原始记录', 'true' if eligible[i] else 'false'),
        ('同日行情有效性', '通过' if valid_bar else '无效行情或无成交量：不产生权益目标'),
        ('资格 / 门禁', ('通过' if effective_eligible else '未通过') + ' / ' + ('通过' if gates[i] else '未通过')),
        ('有效 eligible（含权息隔离）', 'true' if effective_eligible else 'false'),
        ('权息账户隔离', '全期禁止买入，保留本金；风险上限 0' if action_blocked else '未触发'),
        ('年化波动率 σ / 日 ES', _percent(vol[i]) + ' / ' + _percent(es[i])),
        ('信号风险上限（约束前）', _percent(cap)),
        ('最后 5 市场日截止窗口', '已进入：禁止买入，执行目标 0' if deadline else '未进入'),
    ], 'overview')
    rows = [(_text(index + 1), _text(name), f'{int(vote):+d}' if vote else '0')
            for index, (name, vote) in enumerate(zip(OBJECTIVE_NAMES, votes[i]))]
    return ('<h3>同日信号与风险预算</h3>' + details
            + ('<p class="method-note">冻结特征/前沿与权息隔离不一致，请核验并重新生成运行产物；'
               '原始记录仅诊断展示，不重算或改写前沿，有效资格与风险上限已禁止权益。</p>'
               if action_blocked and (eligible[i] or layer == 1) else '')
            + '<p class="hint">风险上限不是已成交仓位，也不是下一笔指令。回撤与冷静期不能由期末汇总重建；'
            '实际目标及成交只认下面账本，不以末日持仓反推历史订单。+1 看多、−1 看空、0 中性；34 行独立展示，不求和。</p>'
            + '<h3>34 维原子投票</h3>' + _table(('#', '独立维度', '投票'), rows))


def _ledger_table(entries):
    labels = dict(build='建仓', add='增持', reduce='减持', exit='退出', unfilled='未成交', action='权息')
    rows = []
    for entry in entries:
        event = entry['event']
        reason = _text(entry.get('reason'))
        if entry.get('block_reason'):
            reason += '；' + _text(entry['block_reason'])
        rows.append((_text(entry['day']), _text(entry.get('signal_day')),
                     _text(labels.get(event, event)) + ' (' + _text(event) + ')', reason,
                     _decimal(entry.get('quantity')), _decimal(entry.get('price')),
                     _decimal(entry.get('gross')), _decimal(entry.get('fee')),
                     _decimal(entry.get('cash')), _decimal(entry.get('shares')),
                                         _percent(entry.get('target_weight')), _decimal(entry.get('cash_per_share')),
                                         _text(entry.get('side')), _decimal(entry.get('raw_price')),
                                         _decimal(entry.get('commission')), _decimal(entry.get('stamp_tax')),
                                         _decimal(entry.get('transfer_fee')), _decimal(entry.get('share_multiplier')),
                                         _decimal(entry.get('eligible_shares')), _text(entry.get('fractional_entitlement')),
                                         _text(entry.get('fractional_sale_approximation'))))
    return _table(('执行/事件日', '信号日', '事件', '原因', '数量（股）', '成交价（元）',
                                     '成交额/权息现金（元）', '费用（元）', '现金（元）', '持股', '目标仓位', '分红（元/股）',
                                     '买卖方向', '原始执行参考价（元）', '佣金（元）', '印花税（元）', '过户费（元）',
                                     '送转倍数', '权息前有权股份', '零碎权益近似', '零碎出售近似'), rows)


def _ledgers(folder, code):
    entries = _json(folder, f'ledgers/{code}.json')
    if not isinstance(entries, list):
        raise ReportError('本地账户账本格式不正确。')
    for entry in entries:
        if not {'day', 'signal_day', 'event', 'reason', 'quantity', 'price', 'gross',
                'fee', 'cash', 'shares', 'target_weight'} <= entry.keys():
            raise ReportError('本地账户账本缺少必需字段。')
        day = _iso(entry['day'])
        if entry['signal_day'] is not None and _iso(entry['signal_day']) > day:
            raise ReportError('账本信号日晚于执行日。')
        for key in ('quantity', 'gross', 'fee', 'cash', 'shares'):
            _number(entry[key])
    fills = [e for e in entries if e['event'] in ('build', 'add', 'reduce', 'exit')]
    others = [e for e in entries if e['event'] not in ('build', 'add', 'reduce', 'exit')]
    return (f'<h3>成交账本：共 {len(fills)} 次，展示最近 {min(100, len(fills))} 次</h3>'
            + _ledger_table(fills[-100:])
            + f'<h3>未成交 / 权息等事件：共 {len(others)} 次，展示最近 {min(100, len(others))} 次</h3>'
            + _ledger_table(others[-100:]))


def _stock(folder, manifest, summary, risk, config, code):
    if code not in [row['code'] for row in manifest['universe']]:
        raise ReportError('未知股票：该代码不在本次完成回测的股票池中；不联网补取或回落旧策略。')
    account = _load_account(folder, code, config, manifest['effective_end'])
    rows = [
        ('初始本金', _money(account['initial_capital'])),
        ('期末现金', _money(account['final_cash'])),
        ('残余持股 / 标记残值', _decimal(account['shares']) + ' 股 / ' + _money(account['residual_value'])),
        ('期末权益 = 现金 + 残值', _money(account['marked_nav'])),
        ('总盈亏 = 权益 − 本金', _money(account['pnl'])),
        ('现金 − 本金（不是含残值总盈亏）', _money(account['final_cash'] - account['initial_capital'])),
        ('已计费用 / 现金分红', _money(account['fees']) + ' / ' + _money(account['cash_dividends'])),
        ('清仓状态', '已全部清仓' if account['liquidation_complete'] else '未完成清仓；残值不是现金'),
        ('个股缓存末日 / 最后有效标记价格', _text(account['last_date']) + ' / ' + _decimal(account['last_price'])),
        ('账户回放截止 / 缓存首日', _text(account['end']) + ' / ' + _text(account['first_date'])),
        ('建仓 / 增持 / 减持 / 退出次数', ' / '.join(_text(account[k + '_count']) for k in ('build', 'add', 'reduce', 'exit'))),
        ('未成交尝试次数 / 权息事件数', _text(account['unfilled_count']) + ' / ' + _text(account['action_count'])),
        ('观察到的最大净值回撤', _percent(account['max_drawdown_observed'])),
        ('零碎股份近似', '是' if account['fractional_share_approximation'] else '否'),
        ('训练末权益 / 验证末权益', _money(account['train_nav']) + ' / ' + _money(account['validation_nav'])),
        ('平均仓位', _percent(account['average_exposure'])),
        ('数据质量', _quality(account)),
    ]
    annual = []
    previous = _number(account['initial_capital'])
    for year, record in sorted(account['years'].items()):
        nav = _number(record['nav'])
        annual.append((_text(year), _text(record['date']), _money(nav), _money(nav - previous),
                       _percent(nav / previous - 1) if previous else '—',
                       _money(record['cash']), _decimal(record['shares'])))
        previous = nav
    return (_header(manifest, f'{account["name"]} ({code}) — 多维融合（帕累托＋风险预算）')
            + _table(('独立账户', '金额 / 状态'), rows, 'overview')
            + '<p class="hint">费用已扣、现金分红已入账，不再叠加到权益；含残值盈亏不等于已兑现现金收益。</p>'
            + _snapshot(folder, manifest, account, risk)
            + '<h3>年度收益</h3><p class="hint">按账户年末记录相对前一记录计算，首笔相对初始本金；'
            '年度有缺口时不插补、不冒充连续单年收益。末年可能不足一年，收益包含标记残值。</p>'
            + _table(('年份', '记录日期', '权益', '本期盈亏', '本期收益率', '现金', '股份'), annual)
            + _ledgers(folder, code) + render_methodology(risk, config) + _limitations(manifest, summary))


def _population(folder, manifest, summary, risk, config, n):
    totals = summary['totals']
    rows = [(label, _money(totals[key])) for key, label in (
        ('initial_capital', '总初始本金'), ('final_cash', '期末总现金'),
        ('residual_marked_value', '残余股份标记总值（非现金）'),
        ('final_equity', '期末总权益 = 现金 + 残值'), ('pnl', '总盈亏 = 权益 − 全部本金'),
        ('fees', '已计费用'), ('cash_dividends', '已入账现金分红'))]
    rows += [('现金 − 全部本金', _money(totals['cash_minus_all_principal'])),
             ('账户总数 / 完全清仓数', _text(totals['account_count']) + ' / ' + _text(totals['fully_liquidated_count'])),
             ('盈利 / 亏损账户数', _text(totals['profitable_accounts']) + ' / ' + _text(totals['losing_accounts'])),
             ('建仓 / 增持 / 减持 / 退出次数', ' / '.join(_text(totals[k + '_count']) for k in ('build', 'add', 'reduce', 'exit'))),
             ('2024 年起留出期盈亏', _money(summary['holdout_2024_onward_pnl']))]
    quality = [(label, _text(totals[key])) for key, label in (
        ('action_fetch_failures', '权息失败或未核验账户（全期禁买，保留本金）'),
        ('rights_issue_accounts', '配股历史账户（全期禁买，保留本金）'),
        ('feature_errors', '特征错误账户'), ('unexplained_gap_accounts', '未解释跳空账户'),
        ('stale_accounts', '陈旧报价账户'), ('late_start_accounts', '晚起始账户'))]
    annual = [(_text(r['year']), _text(r['date']), _money(r['final_equity']), _money(r['pnl']),
               f'{_number(r["return_pct"]):.2f}%') for r in summary['annual_results']]
    accounts = []
    for item in sorted(manifest['universe'], key=lambda item: item['code'])[:n]:
        a = _load_account(folder, item['code'], config, manifest['effective_end'])
        accounts.append((_text(a['code']), _text(a['name']), _money(a['initial_capital']),
                         _money(a['final_cash']), _money(a['residual_value']), _money(a['marked_nav']),
                         _money(a['pnl']), '是' if a['liquidation_complete'] else '否',
                         _text(a['last_date']), _quality(a)))
    trials = [(_percent(t['vol_target']), _money(t['train_pnl']), _money(t['validation_pnl']),
               _money(t['validation_end_nav']), _text(t['scope_end']),
               '所选参数' if t['vol_target'] == risk.annual_vol_target else '—')
              for t in summary['trials']]
    return (_header(manifest, '融合策略回测总览（全部独立账户）')
            + _table(('总资金口径', '完成运行汇总'), rows, 'overview')
            + '<h3>全量数据质量与保守隔离</h3>' + _table(('类别（可重叠，不相加）', '账户数'), quality)
            + '<p class="hint">以上资金及年度收益来自全量 summary，费用与现金分红已计入权益，不重复相加；'
            '未平仓残值不算已兑现现金。年度缺口不插补，以相邻已记录年末权益计算；末年可能不足一年。</p><h3>全量年度收益（非仅年化）</h3>'
            + _table(('年份', '记录日期', '期末权益', '本期盈亏', '本期收益率'), annual)
            + f'<h3>按代码顺序展示前 {len(accounts)} 份账户</h3><p class="method-note">N 仅为展示截断，'
            f'不是按收益排名、加权选择或推荐 Top-N；实际全部 {_text(totals["account_count"])} 个账户均参与，'
            '总资金不随展示数量变化。</p>'
            + _table(('代码', '股票', '本金', '现金', '残值', '权益', '盈亏', '清仓完成', '个股缓存末日', '数据质量'), accounts, 'scan-table')
            + '<h3>风险参数试验（不是股票排名）</h3>'
            + _table(('年化波动目标', '训练期盈亏', '验证期盈亏', '验证末权益', '完整回放截止', '训练期选优'), trials)
            + render_methodology(risk, config) + _limitations(manifest, summary))


def _render(view, *args):
    try:
        content = view(*_load_run(), *args)
        return '<div class="result pareto-report">' + content + '</div>'
    except ReportError as error:
        message = str(error)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, EOFError, BadZipFile):
        message = '本地多维回测产物不完整或无法读取，请等待运行完成并核验数据契约；不联网、不写数据库、不回落旧策略。'
    return '<div class="result pareto-report"><p class="method-note">' + _text(message) + '</p></div>'


def render_stock_report(code):
    if not isinstance(code, str) or not re.fullmatch(r'[0-9]{6}', code):
        return '<div class="error">请输入6位股票代码</div>'
    return _render(_stock, code)


def render_population(n=5):
    try:
        n = max(1, min(100, int(n)))
    except (TypeError, ValueError, OverflowError):
        n = 5
    return _render(_population, n)