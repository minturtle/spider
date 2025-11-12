#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 Spider Execution Accuracy 평가기
- Parsing 없이 원문 SQL 실행
- 열은 모두 정렬해서 비교
- 행은 Gold SQL에 ORDER BY가 있는 경우에만 엄격하게 검사
- 난이도 분류 포함 (evaluation.py와 동일)
- 결과를 JSON 파일로 저장
"""

import os
import sys
import json
import sqlite3
import re
from typing import Dict, List, Tuple, Any

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 난이도 측정을 위한 파싱 (evaluation.py에서 가져옴)
from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql

# evaluation.py의 난이도 측정 함수들
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count


def eval_hardness(sql):
    """evaluation.py와 동일한 난이도 측정"""
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"


def has_order_by(sql_str: str) -> bool:
    """SQL 문자열에 ORDER BY가 있는지 확인"""
    # 대소문자 무시하고 ORDER BY 패턴 찾기
    pattern = r'\bORDER\s+BY\b'
    return bool(re.search(pattern, sql_str, re.IGNORECASE))


def execute_sql(db_path: str, sql: str) -> Tuple[bool, List[Tuple]]:
    """SQL 실행 및 결과 반환"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, []


def normalize_value(val):
    """값을 정규화 (비교를 위해)"""
    if val is None:
        return (2, None)  # None은 (2, None)
    # 숫자와 문자열을 구분해서 정렬 가능하도록 변환
    if isinstance(val, (int, float)):
        return (0, val)  # 숫자는 (0, 값)
    else:
        return (1, str(val))  # 문자열은 (1, 값)


def compare_results(pred_res: List[Tuple], gold_res: List[Tuple], 
                   has_order: bool) -> bool:
    """결과 비교"""
    if len(pred_res) != len(gold_res):
        return False
    
    if len(pred_res) == 0:
        return True
    
    # 열 정렬 (각 행의 컬럼들을 정규화 후 정렬)
    def normalize_row(row):
        normalized = [normalize_value(val) for val in row]
        return tuple(sorted(normalized))
    
    pred_sorted = [normalize_row(row) for row in pred_res]
    gold_sorted = [normalize_row(row) for row in gold_res]
    
    # 행 비교
    if has_order:
        # ORDER BY가 있으면 순서 엄격하게 검사
        return pred_sorted == gold_sorted
    else:
        # ORDER BY가 없으면 순서 무시하고 정렬해서 비교
        return sorted(pred_sorted) == sorted(gold_sorted)


def evaluate_exec(gold_file: str, pred_file: str, db_dir: str, 
                 output_json: str = None) -> Dict[str, Any]:
    """
    Execution Accuracy 평가
    
    Args:
        gold_file: 정답 SQL 파일 (SQL\\tdb_id 형식)
        pred_file: 예측 SQL 파일 (SQL\\tdb_id 형식)
        db_dir: 데이터베이스 디렉토리 경로
        output_json: 결과를 저장할 JSON 파일 경로 (선택)
    
    Returns:
        평가 결과 딕셔너리
    """
    # 파일 읽기
    with open(gold_file, 'r', encoding='utf-8') as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    
    if len(plist) != len(glist):
        print(f"경고: 예측 파일({len(plist)}줄)과 정답 파일({len(glist)}줄)의 줄 수가 다릅니다.")
    
    # 난이도별 통계
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    scores = {level: {'count': 0, 'correct': 0, 'exec_error': 0} for level in levels}
    
    # 상세 결과
    detailed_results = []
    
    total = len(glist)
    print(f"총 쿼리 수: {total}")
    print(f"평가 시작...\n")
    
    for idx, (p, g) in enumerate(zip(plist, glist), 1):
        if len(p) < 1 or len(g) < 2:
            continue
        
        pred_sql = p[0]
        gold_sql = g[0]
        db_name = g[1]
        
        db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
        
        if not os.path.exists(db_path):
            print(f"경고: 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
            continue
        
        # 난이도 측정 (Gold SQL 파싱 필요)
        try:
            schema = Schema(get_schema(db_path))
            g_sql = get_sql(schema, gold_sql)
            hardness = eval_hardness(g_sql)
        except Exception as e:
            # 파싱 실패 시 'all'로 분류
            hardness = 'all'
        
        scores[hardness]['count'] += 1
        scores['all']['count'] += 1
        
        # SQL 실행
        pred_success, pred_res = execute_sql(db_path, pred_sql)
        gold_success, gold_res = execute_sql(db_path, gold_sql)
        
        if not pred_success:
            scores[hardness]['exec_error'] += 1
            scores['all']['exec_error'] += 1
            detailed_results.append({
                'idx': idx,
                'hardness': hardness,
                'pred_sql': pred_sql,
                'gold_sql': gold_sql,
                'db_name': db_name,
                'correct': False,
                'error': 'pred_exec_error'
            })
            continue
        
        if not gold_success:
            scores[hardness]['exec_error'] += 1
            scores['all']['exec_error'] += 1
            detailed_results.append({
                'idx': idx,
                'hardness': hardness,
                'pred_sql': pred_sql,
                'gold_sql': gold_sql,
                'db_name': db_name,
                'correct': False,
                'error': 'gold_exec_error'
            })
            continue
        
        # ORDER BY 체크
        has_order = has_order_by(gold_sql)
        
        # 결과 비교
        is_correct = compare_results(pred_res, gold_res, has_order)
        
        if is_correct:
            scores[hardness]['correct'] += 1
            scores['all']['correct'] += 1
        
        detailed_results.append({
            'idx': idx,
            'hardness': hardness,
            'pred_sql': pred_sql,
            'gold_sql': gold_sql,
            'db_name': db_name,
            'correct': is_correct,
            'has_order_by': has_order,
            'pred_result_count': len(pred_res),
            'gold_result_count': len(gold_res)
        })
        
        if idx % 100 == 0:
            current_acc = scores['all']['correct'] / scores['all']['count']
            print(f"진행 중... {idx}/{total} (정확도: {current_acc:.3f})")
    
    # 정확도 계산
    results = {
        'summary': {},
        'detailed': detailed_results
    }
    
    for level in levels:
        if scores[level]['count'] > 0:
            accuracy = scores[level]['correct'] / scores[level]['count']
            results['summary'][level] = {
                'count': scores[level]['count'],
                'correct': scores[level]['correct'],
                'exec_error': scores[level]['exec_error'],
                'accuracy': accuracy
            }
        else:
            results['summary'][level] = {
                'count': 0,
                'correct': 0,
                'exec_error': 0,
                'accuracy': 0.0
            }
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("Execution Accuracy 결과")
    print("=" * 60)
    print(f"{'난이도':<10} {'Count':<10} {'Correct':<10} {'Error':<10} {'Accuracy':<10}")
    print("-" * 60)
    for level in levels:
        s = results['summary'][level]
        print(f"{level:<10} {s['count']:<10} {s['correct']:<10} {s['exec_error']:<10} {s['accuracy']:.3f}")
    print("=" * 60)
    
    # JSON 파일로 저장
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n결과가 저장되었습니다: {output_json}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Spider Execution Accuracy 평가기')
    parser.add_argument('--gold', type=str, required=True, help='정답 SQL 파일 경로')
    parser.add_argument('--pred', type=str, required=True, help='예측 SQL 파일 경로')
    parser.add_argument('--db', type=str, required=True, help='데이터베이스 디렉토리 경로')
    parser.add_argument('--output', type=str, default=None, help='결과 JSON 파일 경로 (선택)')
    
    args = parser.parse_args()
    
    evaluate_exec(args.gold, args.pred, args.db, args.output)

