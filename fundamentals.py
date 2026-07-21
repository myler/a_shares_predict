#!/usr/bin/env python3
"""巴芒基本面研究的财务质量初筛。

本模块只处理公开财务摘要中的客观字段；不推断护城河、管理层品质或内在价值，
也不产生买卖信号。
"""

from __future__ import annotations

from statistics import median


STATUS_GO = 'go'
STATUS_WATCH = 'watch'
STATUS_NO_GO = 'no_go'
STATUS_INSUFFICIENT = 'insufficient'
STATUS_NOT_APPLICABLE = 'not_applicable'

_FINANCIAL_FIELDS = (
    'TOTALDEPOSITS', 'GROSSLOANS', 'NET_INTEREST_SPREAD',
    'NET_INTEREST_MARGIN', 'SOLVENCY_AR', 'NET_CAPITAL_LIABILITIES',
)


def _number(value):
    if isinstance(value, bool) or value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_key(value):
    return str(value or '')[:10]


def _has_value(value):
    number = _number(value)
    return number is not None and number != 0


def _annual_reports(rows):
    reports = []
    seen_dates = set()
    for row in sorted(rows or [], key=lambda item: _date_key(
            item.get('REPORT_DATE')), reverse=True):
        if '年报' not in str(row.get('REPORT_TYPE') or ''):
            continue
        report_date = _date_key(row.get('REPORT_DATE'))
        if not report_date or report_date in seen_dates:
            continue
        seen_dates.add(report_date)
        reports.append(row)
    return reports


def _check(key, label, status, detail):
    return {'key': key, 'label': label, 'status': status, 'detail': detail}


def _value_detail(value, unit='', digits=1):
    if value is None:
        return '数据缺失'
    return f'{value:.{digits}f}{unit}'


def _is_financial_institution(reports):
    for report in reports:
        for field in _FINANCIAL_FIELDS:
            if _has_value(report.get(field)):
                return True
    return False


def _normalise_report(report):
    return {
        'report_date': _date_key(report.get('REPORT_DATE')),
        'report_name': str(report.get('REPORT_DATE_NAME') or report.get('REPORT_TYPE') or ''),
        'notice_date': _date_key(report.get('NOTICE_DATE')),
        'currency': str(report.get('CURRENCY') or 'CNY'),
        'revenue': _number(report.get('TOTALOPERATEREVE')),
        'parent_net_profit': _number(report.get('PARENTNETPROFIT')),
        'core_profit': _number(report.get('KCFJCXSYJLR')),
        'operating_cash_flow': _number(report.get('NETCASH_OPERATE_PK')),
        'roe': _number(report.get('ROEJQ')),
        'roic': _number(report.get('ROIC')),
        'debt_ratio': _number(report.get('ZCFZL')),
        'current_ratio': _number(report.get('LD')),
    }


def screen_value_quality(rows):
    """从财务摘要生成可解释的非金融企业财务质量初筛结果。

    ``rows`` 必须是按任意顺序传入的东方财富财务摘要原始行。结果中的金额保持
    上游的 CNY 原始单位，调用者应在展示层再统一换算。
    """
    annuals = _annual_reports(rows)
    reports = [_normalise_report(report) for report in annuals]
    company = str((annuals[0] if annuals else {}).get('SECURITY_NAME_ABBR') or '')
    code = str((annuals[0] if annuals else {}).get('SECURITY_CODE') or '')
    base = {
        'company': company,
        'code': code,
        'annual_reports': reports,
        'source': '东方财富财务摘要',
        'scope': '非金融企业财务质量初筛；不替代年报原文核验、护城河/治理审查或估值。',
        'limitations': [
            '未核验审计意见、财报附注、关联交易、表外负债和管理层治理。',
            '未判断行业周期、竞争优势、维护性资本开支或内在价值。',
            '财务摘要字段须在进入正式估值前回到公司正式披露文件复核。',
        ],
    }

    if _is_financial_institution(annuals):
        return {
            **base,
            'status': STATUS_NOT_APPLICABLE,
            'status_label': '方法不适用',
            'conclusion': '检测到金融机构字段；银行、保险、券商等需要专用的资本与估值框架。',
            'checks': [_check('industry_scope', '行业适用性', 'unavailable',
                              '检测到存贷款、净息差或偿付能力等金融机构字段')],
        }

    if len(reports) < 3:
        return {
            **base,
            'status': STATUS_INSUFFICIENT,
            'status_label': '数据不足',
            'conclusion': '可用年报少于三份，不能建立最低限度的盈利与现金流连续性判断。',
            'checks': [_check('history', '年报连续性', 'unavailable',
                              f'仅取得 {len(reports)} 份去重年报，至少需要 3 份')],
        }

    recent = reports[:3]
    checks = []
    core_profits = [report['core_profit'] for report in recent]
    if any(value is None for value in core_profits):
        checks.append(_check('profitability', '扣非盈利连续性', 'unavailable',
                             '最近三份年报存在扣非归母净利润缺失'))
    elif core_profits[0] <= 0 or sum(value <= 0 for value in core_profits) >= 2:
        checks.append(_check('profitability', '扣非盈利连续性', 'fail',
                             '最新年报扣非归母净利润为负，或近三年出现至少两次亏损'))
    elif all(value > 0 for value in core_profits):
        checks.append(_check('profitability', '扣非盈利连续性', 'pass',
                             '最近三份年报的扣非归母净利润均为正'))
    else:
        checks.append(_check('profitability', '扣非盈利连续性', 'watch',
                             '近三年存在一次扣非亏损，需核验其是否一次性或结构性问题'))

    operating_cash_flows = [report['operating_cash_flow'] for report in recent]
    if any(value is None for value in operating_cash_flows) or any(
            value is None for value in core_profits):
        checks.append(_check('cash_conversion', '经营现金流转化', 'unavailable',
                             '最近三年经营现金流或扣非利润数据缺失'))
    else:
        total_core_profit = sum(core_profits)
        conversion = sum(operating_cash_flows) / total_core_profit \
            if total_core_profit > 0 else None
        if conversion is None:
            checks.append(_check('cash_conversion', '经营现金流转化', 'watch',
                                 '近三年累计扣非利润不为正，无法计算有效现金转化率'))
        elif conversion >= 0.8:
            checks.append(_check('cash_conversion', '经营现金流转化', 'pass',
                                 f'近三年累计经营现金流/扣非利润为 {conversion:.2f}'))
        else:
            checks.append(_check('cash_conversion', '经营现金流转化', 'watch',
                                 f'近三年累计经营现金流/扣非利润为 {conversion:.2f}，需核验营运资本与应收项'))

    roes = [report['roe'] for report in recent if report['roe'] is not None]
    if len(roes) < 3:
        checks.append(_check('roe', 'ROE持续性', 'unavailable',
                             '最近三份年报的加权ROE数据不完整'))
    else:
        median_roe = median(roes)
        if median_roe >= 10:
            checks.append(_check('roe', 'ROE持续性', 'pass',
                                 f'近三年加权ROE中位数为 {median_roe:.1f}%'))
        elif median_roe >= 5:
            checks.append(_check('roe', 'ROE持续性', 'watch',
                                 f'近三年加权ROE中位数为 {median_roe:.1f}%，未达到初筛目标 10%'))
        else:
            checks.append(_check('roe', 'ROE持续性', 'fail',
                                 f'近三年加权ROE中位数为 {median_roe:.1f}%，盈利资本回报偏低'))

    latest = recent[0]
    debt_ratio = latest['debt_ratio']
    current_ratio = latest['current_ratio']
    if debt_ratio is None:
        checks.append(_check('leverage', '杠杆与流动性', 'unavailable',
                             '最新年报资产负债率数据缺失'))
    elif debt_ratio > 80 and current_ratio is not None and current_ratio < 1:
        checks.append(_check('leverage', '杠杆与流动性', 'fail',
                             f'资产负债率 {debt_ratio:.1f}% 且流动比率 {current_ratio:.2f} < 1'))
    elif debt_ratio <= 70 and (current_ratio is None or current_ratio >= 1):
        ratio_note = _value_detail(current_ratio, digits=2)
        checks.append(_check('leverage', '杠杆与流动性', 'pass',
                             f'资产负债率 {debt_ratio:.1f}%，流动比率 {ratio_note}'))
    else:
        ratio_note = _value_detail(current_ratio, digits=2)
        checks.append(_check('leverage', '杠杆与流动性', 'watch',
                             f'资产负债率 {debt_ratio:.1f}%，流动比率 {ratio_note}，需按行业复核'))

    revenue_values = [report['revenue'] for report in recent]
    if all(value is not None for value in revenue_values) and revenue_values[-1] > 0:
        revenue_change = revenue_values[0] / revenue_values[-1] - 1
        trend_status = 'pass' if revenue_change >= 0 else 'watch'
        trend_detail = f'最近三份年报收入累计变化 {revenue_change:+.1%}'
    else:
        trend_status = 'unavailable'
        trend_detail = '最近三份年报收入数据不完整'
    checks.append(_check('revenue_trend', '收入规模趋势', trend_status, trend_detail))

    check_statuses = {item['key']: item['status'] for item in checks}
    critical = ('profitability', 'cash_conversion', 'roe', 'leverage')
    if any(check_statuses.get(key) == 'fail' for key in critical):
        status = STATUS_NO_GO
        status_label = '暂不进入价值候选池'
        conclusion = '核心财务质量项目存在红线或明显不足；这不是卖出信号，而是暂不开展估值的研究结论。'
    elif all(check_statuses.get(key) == 'pass' for key in critical):
        status = STATUS_GO
        status_label = '通过财务质量初筛'
        conclusion = '基础盈利、现金转化、资本回报和杠杆均通过初筛；仍须完成行业、护城河、治理与原始年报复核。'
    else:
        status = STATUS_WATCH
        status_label = '观察并补充核验'
        conclusion = '未发现自动否决项，但关键质量证据不足或存在需解释的信号，不进入估值或交易结论。'

    return {
        **base,
        'status': status,
        'status_label': status_label,
        'conclusion': conclusion,
        'checks': checks,
    }