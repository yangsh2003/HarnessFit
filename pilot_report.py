#!/usr/bin/env python3
"""Build a Chinese pilot Markdown report from real runner artifacts; optional grid SVG."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

LABELS = {
    'resolved': '官方解决', 'unresolved': '官方未解决', 'empty_patch': '空补丁（已判）',
    'empty_pending': '空补丁（待判）', 'pending': '有补丁待判', 'infra': '运行／判题异常',
    'invalid': '记录不完整或不匹配',
}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


def one_line(value, limit=180):
    text = ' '.join(str(value).split())
    if len(text) > limit:
        text = text[:limit-1] + '…'
    return (escape(text).replace('|', '&#124;').replace('`', '&#96;')
            .replace('[', '&#91;').replace(']', '&#93;'))


def link(path, output):
    return quote(os.path.relpath(path.resolve(), output.parent.resolve()), safe='/')


def inspect_slot(slot, root):
    model, cfg, seed, issue = slot.relative_to(root).parts
    record = dict(model=model, cfg=cfg, seed=int(seed), issue=issue, slot=slot,
                  state='invalid', note='', r={}, patch='', verdict=None, report=None)
    try:
        r = read(slot/'result.json')
        record['r'] = r
        if (r['model'], r['config'], r['seed'], r['instance_id']) != (model, cfg, int(seed), issue):
            raise ValueError('结果身份与目录不一致')
        if r['status'] == 'infra_error':
            record.update(state='infra', note=r.get('error','infra_error'))
            return record
        if not all(isinstance(r.get(k), str) and r[k] for k in ('base_commit', 'model_revision', 'image_id', 'protocol_sha256')):
            raise ValueError('缺少版本或镜像信息')
        fp = digest(slot/'prediction.json')
        if r.get('prediction_sha256') != fp or r.get('trace_sha256') != digest(slot/'trace.jsonl'):
            raise ValueError('运行摘要的 prediction 或 trace 校验值不符')
        pred = read(slot/'prediction.json')
        patch = pred['model_patch']
        if not isinstance(patch, str) or pred['instance_id'] != issue or pred['model_name_or_path'] != model:
            raise ValueError('预测记录或任务身份不符')
        record['patch'] = patch
        label = slot/'verdict.json'
        if not label.exists():
            record['state'] = 'pending' if patch.strip() else 'empty_pending'
            return record
        v = read(label)
        if v['prediction_sha256'] != fp or v['instance_id'] != issue or not v['origin'].startswith('swebench=='):
            raise ValueError('判分摘要与预测或评分器不符')
        report_file = (slot/v['report']).resolve()
        if not report_file.is_relative_to(root.resolve()):
            raise ValueError('判分报告路径越出 runs 目录')
        official = read(report_file)
        state = v['status']
        if state == 'evaluation_error' and v['resolved'] is None:
            record.update(state='infra', note='官方评测未生成有效判决', verdict=v, report=report_file)
            return record
        if state not in ('resolved', 'unresolved', 'empty_patch') or v['resolved'] is not (state == 'resolved'):
            raise ValueError('判分状态无效')
        if issue not in official[state+'_ids'] or bool(patch.strip()) == (state == 'empty_patch'):
            raise ValueError('官方报告、判分标签或补丁状态相互矛盾')
        record.update(state=state, verdict=v, report=report_file)
    except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
        record['note'] = str(exc)
    return record


def choose_case(rows):
    # Same issue, seed, model revision, base, image and run budget for a paired case.
    cases = defaultdict(list)
    for row in rows:
        if row['state'] in ('invalid', 'infra'):
            continue
        r = row['r']
        key = (row['model'], row['issue'], row['seed'], r['base_commit'], r['model_revision'],
               r['image_id'], r['protocol_sha256'])
        cases[key].append(row)
    if not cases:
        return []
    # Prefer the earliest fully judged pair with opposing official labels. Fall
    # back to the earliest fully judged pair, then to a pair or single trajectory.
    choices = []
    for key, group in sorted(cases.items()):
        group = sorted(group, key=lambda r:r['cfg'])
        for i, a in enumerate(group):
            for b in group[i+1:]:
                judged = all(x['verdict'] for x in (a,b))
                different = judged and (a['state']=='resolved') != (b['state']=='resolved')
                choices.append((0 if different else 1 if judged else 2,key,a['cfg'],b['cfg'],[a,b]))
        for a in group:
            choices.append((3 if a['verdict'] else 4,key,a['cfg'],'',[a]))
    return min(choices, key=lambda c:c[:-1])[-1]


def action_sample(row):
    candidate = None
    with (row['slot']/'trace.jsonl').open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            event = json.loads(line)
            if event.get('event') != 'action':
                continue
            tool = event.get('tool')
            if tool:
                payload = event.get('payload', '')
                if isinstance(payload, dict):
                    payload = payload.get('path', str(payload))
                excerpt = f"L{n} · step {event.get('step', '?')} · {tool} {payload}"
            else:
                excerpt = f"L{n} · step {event.get('step', '?')} · {event.get('observation', '')}"
            if event.get('changed'):
                return excerpt
            if candidate is None:
                candidate = excerpt
    return candidate or '轨迹未出现 action 事件'


def heatmap(study, split, dest):
    plan = read(study/'plan.json')
    configs = plan['configs']
    models = [plan['baseline'],plan['target']] + plan['api_models']
    grid = {}
    # study.py report --full writes the common table after checking all model
    # grids. For an earlier single-model grid, read its audited JSON instead.
    complete = study/f'heatmap-{split}.csv'
    shared = {}
    if complete.is_file():
        with complete.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                m, c = row['model'], row['config']
                if m not in models or row['split'] != split or c in shared.get(m, {}):
                    raise ValueError(f'汇总表有重复单元或与 plan 不匹配: {complete}')
                seeds = plan['seeds'][:1] if m in plan['api_models'] else plan['seeds']
                if list(map(int, row['seeds'].split(','))) != seeds:
                    raise ValueError(f'汇总表的 seed 与 plan 不符: {m}/{c}')
                shared.setdefault(m,{})[c] = dict(attempts=int(row['attempts']), solved=int(row['solved']),
                                                  rate=float(row['resolve_rate']))
        if set(shared) != set(models):
            raise ValueError(f'完整表缺少预注册模型: {complete}')
    for m in models:
        path = study/f'grid-{m}-{split}.json'
        if m in shared:
            path = complete
            data = dict(model=m, split=split, configs=shared[m])
        elif path.is_file():
            data = read(path)
        else:
            continue
        cells = data['configs']
        expected = plan['api_configs'] if m in plan['api_models'] else configs
        n_seeds = 1 if m in plan['api_models'] else len(plan['seeds'])
        n = len(plan[split])*n_seeds
        if data['split'] != split or data['model'] != m or set(cells) != set(expected) or n < 1:
            raise ValueError(f'网格与 plan 不匹配: {path}')
        for c in expected:
            item = cells[c]
            if item['attempts'] != n or not 0 <= item['solved'] <= n or abs(item['solved']/n-item['rate']) > 1e-9:
                raise ValueError(f'网格分母或解决率不符: {path} {c}')
        grid[m] = cells
    if not grid:
        raise ValueError(f'缺少已完成的 grid-模型-{split}.json，先运行 study.py grid')
    width, left, cell, row_height = 1100, 220, 53, 46
    height = 140 + row_height*len(grid)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif">',
           f'<text x="20" y="35" font-size="24">Harness resolve rate · {escape(split)} · {len(plan[split])} issues</text>']
    for j,c in enumerate(configs):
        svg.append(f'<text x="{left+j*cell+cell/2}" y="77" text-anchor="middle" font-size="12">{escape(c)}</text>')
    for i,(m,cells) in enumerate(grid.items()):
        y = 85 + i*row_height
        svg.append(f'<text x="20" y="{y+26}" font-size="16">{escape(m[:23])}</text>')
        for j,c in enumerate(configs):
            x = left+j*cell
            if c in cells:
                item = cells[c]
                rate = item['rate']
                rgb = [round(a+(b-a)*rate) for a,b in zip((237,246,244),(7,92,81))]
                bg = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
                fg = 'white' if rate > .58 else '#173432'
                val, title = f'{100*rate:.1f}%', f"{m}/{c}: {item['solved']}/{item['attempts']}"
            else:
                bg, fg, val, title = '#f1f1f1','#656565','—','outside registered API grid'
            svg.append(f'<g><title>{escape(title)}</title><rect x="{x}" y="{y}" width="{cell-2}" height="{row_height-3}" fill="{bg}"/>'
                       f'<text x="{x+cell/2}" y="{y+26}" fill="{fg}" text-anchor="middle" font-size="13">{val}</text></g>')
    svg.append(f'<text x="20" y="{height-15}" font-size="13">Open models: {len(plan["seeds"])} seeds. API: one seed, shared subset. Only audited complete model grids are shown.</text></g></svg>')
    dest.write_text('\n'.join(svg)+'\n',encoding='utf-8')
    return list(grid)


def main():
    p = argparse.ArgumentParser(description='将独立 9B pilot 的真实运行记录导出为中文 Markdown；可选导出已完成网格的 SVG 热图。')
    p.add_argument('--runs', type=Path, required=True, help='pilot 的 runs 根目录，层级 MODEL/CONFIG/SEED/INSTANCE')
    p.add_argument('--model', default='qwen9b', help='模型服务别名，默认 qwen9b')
    p.add_argument('--out', type=Path, default=Path('pilot-report.md'), help='输出的 Markdown 文件；证据链接相对于此文件')
    p.add_argument('--study', type=Path, help='可选，study.py 预注册目录，应有 plan.json 和完整 grid JSON 或 heatmap CSV')
    p.add_argument('--split', choices=['dev','holdout'], default='dev', help='SVG 使用的正式网格切分，默认 dev')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.model):
        p.error('无效模型别名')
    slots = sorted(x for x in (args.runs/args.model).glob('*/*/*') if x.is_dir()
                   and not x.name.startswith('.') and x.parent.name.isdigit()
                   and re.fullmatch('[01]{5}', x.parent.parent.name))
    if not slots:
        p.error('找不到所选模型的运行目录')
    rows = [inspect_slot(slot,args.runs) for slot in slots]
    cfgs = sorted({r['cfg'] for r in rows})
    out = ['# SWE-bench Verified mini · 9B 先导记录', '',
           f"服务别名 `{one_line(args.model)}`，已发现 {len(rows)} 次运行、{len({r['issue'] for r in rows})} 个不同 issue。各行统计单位为单个配置、issue 和 seed 的一次尝试。尚未创建的运行目录没有计入。",
           '', '|配置|运行数|'+'|'.join(LABELS.values())+'|', '|---|'+'---:|'*(1+len(LABELS))]
    for c in cfgs:
        group = [x for x in rows if x['cfg']==c]
        counts = Counter(r['state'] for r in group)
        out.append(f'|`{c}`|{len(group)}|'+'|'.join(str(counts[k]) for k in LABELS)+'|')
    usable = [r['r'] for r in rows if r['state'] not in ('invalid','infra')]
    diags = [r['diagnostics'] for r in usable if 'diagnostics' in r]
    out.extend(['','## 格式与执行诊断','',
                f"在有完整运行摘要的 {len(diags)} 次非异常尝试中，"
                f"{sum(d.get('parse_errors',0)>0 for d in diags)} 次出现过动作解析错误，"
                f"{sum(d.get('edits_seen') is False for d in diags)} 次未观察到文件改动，"
                f"{sum(d.get('valid_test_runs',0)>0 for d in diags)} 次记录到测试或复现调用。判决以独立官方报告为准。"])
    groups = Counter((r['r'].get('model_revision') or '未记录',
                      r['r'].get('protocol_sha256') or '未记录') for r in rows if r['r'])
    out.extend(['', '|模型 revision|运行协议 SHA256|运行数|','|---|---|---:|'])
    for (revision, protocol),count in sorted(groups.items()):
        out.append(f'|`{one_line(revision)}`|`{one_line(protocol)}`|{count}|')
    if len(groups)>1:
        out.extend(['','先导汇总包含多个模型或协议版本；上表仅用于排错。成对案例仅使用版本与预算匹配的轨迹。'])
    case = choose_case(rows)
    if case:
        r = case[0]
        out.extend(['','## 可用于展示的先导轨迹','',
                    f"实例 `{r['issue']}`，seed {r['seed']}，base commit `{r['r']['base_commit']}`。"
                    f"这里有 {len(case)} 条同条件真实轨迹。优先选择同题同 seed、官方判决不同的一对，再按实例与配置 ID 取首例。",
                    '', '|配置|实际动作摘录|补丁新增行摘录|判分证据|','|---|---|---|---|'])
        for row in case:
            slot = row['slot']
            added = next((s for s in row['patch'].splitlines() if s.startswith('+') and not s.startswith('+++')),'')
            patch = one_line(added or ('没有新增行' if row['patch'] else '空补丁'))
            label = LABELS[row['state']]
            score = f"[{label}]({link(row['report'],args.out)})" if row['report'] else label
            trace = one_line(action_sample(row))
            out.append(f"|`{row['cfg']}`|[{trace}]({link(slot/'trace.jsonl',args.out)})|[{patch}]({link(slot/'prediction.json',args.out)})|{score}|")
    errors = [r for r in rows if r['state'] in ('infra', 'invalid')]
    if errors:
        out.extend(['','## 异常运行',''])
        for row in errors:
            out.append(f"- [`{one_line(row['slot'].relative_to(args.runs))}`]({link(row['slot'],args.out)}) · {one_line(row['note'])}")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.study:
        dest = args.out.with_name(args.out.stem+'-'+args.split+'-heatmap.svg')
        models = heatmap(args.study,args.split,dest)
        out.extend(['','## 正式实验的已完成网格','', f"切分为 {args.split}，包含 {', '.join(one_line(m) for m in models)}。", '',
                    f"![Harness resolve rates]({link(dest,args.out)})"])
        print(dest)
    args.out.write_text('\n'.join(out)+'\n',encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc))
