import os
import sys
import argparse
import json
import io
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import asdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from data_structures import (
    Course, Classroom, TimeSlot, Schedule, ScheduleAssignment,
    create_sample_data, expand_courses_to_sessions,
    create_sample_preferences, create_sample_locks
)
from conflict_detector import (
    ConflictDetector, ConflictReport, Conflict, generate_random_schedule
)
from genetic_scheduler import (
    GeneticAlgorithmScheduler, GAHistory, PreferenceResult, find_adjustments
)
from visualizer import (
    visualize_timetable, visualize_convergence,
    visualize_teacher_distribution, visualize_adjustments,
    visualize_conflict_detailed
)


OUTPUT_DIR = "output"


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║          大学教务排课辅助系统 (Course Scheduler)             ║
    ║                                                              ║
    ║   功能: 冲突检测 + 遗传算法优化 + 可视化输出                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_conflict_report(report: ConflictReport, title: str = "冲突检测报告"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  教师冲突:    {report.teacher_conflicts:3d} 起")
    print(f"  班级冲突:    {report.class_conflicts:3d} 起")
    print(f"  教室冲突:    {report.classroom_conflicts:3d} 起")
    print(f"  容量冲突:    {report.capacity_violations:3d} 起")
    print(f"  类型冲突:    {report.type_violations:3d} 起")
    print(f"  ─────────────────────────────")
    print(f"  总冲突数:    {report.total_conflicts():3d} 起")
    print(f"  状态:        {'✖ 存在冲突' if report.has_conflicts() else '✓ 无冲突'}")

    if report.conflicts:
        print(f"\n  详细冲突列表 (最多显示前10条):")
        for i, c in enumerate(report.conflicts[:10]):
            type_map = {
                "teacher": "教师冲突",
                "class": "班级冲突",
                "classroom": "教室冲突",
                "capacity": "容量冲突",
                "type": "类型冲突",
            }
            t = type_map.get(c.conflict_type, c.conflict_type)
            print(f"    [{i+1:2d}] {t}: {c.detail}")
        if len(report.conflicts) > 10:
            print(f"    ... 另有 {len(report.conflicts) - 10} 条冲突未显示")
    print(f"{'='*60}\n")


def print_adjustments(adjustments, expanded_courses, schedule):
    course_map = {c.course_id: c for c in expanded_courses}
    classroom_map = schedule.get_classroom_map()
    timeslot_map = schedule.get_timeslot_map()

    print(f"\n{'='*60}")
    print(f"  课程调整记录")
    print(f"{'='*60}")

    if not adjustments:
        print("  ✓ 无需调整 - 原课表无冲突!")
    else:
        print(f"  共调整 {len(adjustments)} 门课程:\n")
        print(f"  {'序号':<4} {'课程名称':<18} {'教师':<8} {'原安排':<25} {'→':<3} {'新安排'}")
        print(f"  {'─'*80}")

        for idx, (cid, old_a, new_a) in enumerate(adjustments, 1):
            course = course_map[cid]

            if old_a:
                old_ts = timeslot_map[old_a.timeslot_id]
                old_room = classroom_map[old_a.classroom_id]
                old_str = f"{old_ts.day_name}{old_ts.period_name} {old_room.name}"
            else:
                old_str = "N/A"

            new_ts = timeslot_map[new_a.timeslot_id]
            new_room = classroom_map[new_a.classroom_id]
            new_str = f"{new_ts.day_name}{new_ts.period_name} {new_room.name}"

            print(f"  [{idx:<2}] {course.name:<18} {course.teacher:<8} {old_str:<25} → {new_str}")
    print(f"{'='*60}\n")


def print_preference_result(pref_result: PreferenceResult):
    print(f"\n{'='*60}")
    print(f"  偏好约束满足情况")
    print(f"{'='*60}")
    if pref_result.total_preferences == 0:
        print("  未设置任何偏好约束")
    else:
        print(f"  总偏好数:      {pref_result.total_preferences}")
        print(f"  硬约束数:      {pref_result.hard_preferences}")
        print(f"  软偏好数:      {pref_result.soft_preferences}")
        print(f"  满足数量:      {pref_result.satisfied_count}")
        print(f"  硬约束满足:    {pref_result.hard_satisfied}/{pref_result.hard_preferences}")
        print(f"  软偏好满足:    {pref_result.soft_satisfied}/{pref_result.soft_preferences}")
        print(f"  满足率:        {pref_result.satisfaction_rate*100:.1f}%")

        hard_violated = pref_result.hard_preferences - pref_result.hard_satisfied
        if hard_violated > 0:
            print(f"  ⚠  有 {hard_violated} 条硬约束未能满足!")

        if pref_result.violated_details:
            print(f"\n  未满足偏好详情 (前10条):")
            for i, v in enumerate(pref_result.violated_details[:10]):
                prio = "硬约束" if v["priority"] == "hard" else "软偏好"
                print(f"    [{i+1:2d}] [{prio}] {v['description']}")
                print(f"         课程: {v['course_name']}")
                print(f"         实际安排: 周{v['actual_day']+1} 第{v['actual_period']+1}-{v['actual_period']+2}节")
            if len(pref_result.violated_details) > 10:
                print(f"    ... 另有 {len(pref_result.violated_details) - 10} 条未显示")
    print(f"{'='*60}\n")


def print_timetable_summary(expanded_courses, assignments, schedule):
    course_map = {c.course_id: c for c in expanded_courses}
    classroom_map = schedule.get_classroom_map()
    timeslot_map = schedule.get_timeslot_map()

    print(f"\n{'='*60}")
    print(f"  最终课表一览")
    print(f"{'='*60}")

    day_names = ["周一", "周二", "周三", "周四", "周五"]
    period_names = ["第1-2节", "第3-4节", "第5-6节", "第7-8节", "第9-10节"]

    for day_idx, day_name in enumerate(day_names):
        print(f"\n  ┌─ {day_name} {'─'*(45 - len(day_name.encode('gbk')))}┐")
        day_courses = [a for a in assignments
                       if timeslot_map[a.timeslot_id].day == day_idx]
        if not day_courses:
            print(f"  │  (全天无课)                                   │")
            continue

        day_courses.sort(key=lambda x: timeslot_map[x.timeslot_id].period)
        current_period = -1
        for a in day_courses:
            ts = timeslot_map[a.timeslot_id]
            course = course_map[a.course_id]
            room = classroom_map[a.classroom_id]

            if ts.period != current_period:
                current_period = ts.period
                print(f"  │ {period_names[ts.period]}:")
            classes_str = ','.join(course.classes)
            if len(classes_str) > 12:
                classes_str = classes_str[:10] + ".."
            print(f"  │   · {course.name:<14} {course.teacher:<6} "
                  f"{room.name:<12} [{classes_str}]")
    print(f"\n{'='*60}\n")


def save_results_to_json(
    expanded_courses,
    old_assignments,
    new_assignments,
    adjustments,
    report_before,
    report_after,
    history,
    pref_result,
    output_path
):
    result = {
        "summary": {
            "total_courses": len(expanded_courses),
            "adjusted_count": len(adjustments),
            "conflicts_before": report_before.total_conflicts(),
            "conflicts_after": report_after.total_conflicts(),
            "ga_generations": len(history.generations),
            "ga_elapsed_seconds": round(history.elapsed_time, 3),
            "final_best_fitness": round(history.best_fitness[-1], 6),
            "pref_total": pref_result.total_preferences,
            "pref_satisfied": pref_result.satisfied_count,
            "pref_satisfaction_rate": round(pref_result.satisfaction_rate, 4),
            "hard_pref_satisfied": pref_result.hard_satisfied,
            "hard_pref_total": pref_result.hard_preferences,
        },
        "conflicts_before": asdict(report_before),
        "conflicts_after": asdict(report_after),
        "preferences": {
            "total": pref_result.total_preferences,
            "hard_total": pref_result.hard_preferences,
            "soft_total": pref_result.soft_preferences,
            "satisfied": pref_result.satisfied_count,
            "hard_satisfied": pref_result.hard_satisfied,
            "soft_satisfied": pref_result.soft_satisfied,
            "satisfaction_rate": pref_result.satisfaction_rate,
            "violated_details": pref_result.violated_details,
        },
        "adjustments": [],
    }

    course_map = {c.course_id: c for c in expanded_courses}
    for cid, old_a, new_a in adjustments:
        course = course_map[cid]
        result["adjustments"].append({
            "course_id": cid,
            "course_name": course.name,
            "teacher": course.teacher,
            "classes": course.classes,
            "old": asdict(old_a) if old_a else None,
            "new": asdict(new_a),
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至: {output_path}")


def run_scheduler(
    population_size: int = 80,
    max_generations: int = 300,
    mutation_rate: float = 0.08,
    elite_count: int = 4,
    random_seed: int = 42,
    use_random_initial: bool = False,
    use_improved_ga: bool = True,
    fitness_sharing: bool = True,
    adaptive_mutation: bool = True,
    diversity_preserve: bool = True,
    migration_rate: float = 0.05,
    large_scale_test: bool = False,
    use_preferences: bool = True,
    use_locks: bool = True,
):
    ensure_output_dir()
    print_banner()

    print("  [步骤1/7] 加载排课数据...")
    courses, classrooms, timeslots = create_sample_data()
    schedule = Schedule(
        courses=courses,
        classrooms=classrooms,
        timeslots=timeslots,
        assignments=[]
    )
    print(f"    ✓ 课程数: {len(courses)} | 教室数: {len(classrooms)} | 时间段数: {len(timeslots)}")

    preferences = create_sample_preferences() if use_preferences else []
    locked_courses = create_sample_locks() if use_locks else []
    if preferences:
        print(f"    ✓ 偏好约束: {len(preferences)} 条 (硬约束{sum(1 for p in preferences if p.priority=='hard')}条)")
    if locked_courses:
        print(f"    ✓ 手动锁定: {len(locked_courses)} 门课程")

    print("\n  [步骤2/7] 展开为独立课时...")
    expanded_courses = expand_courses_to_sessions(courses)
    print(f"    ✓ 展开后课程节数: {len(expanded_courses)}")
    for c in expanded_courses:
        valid_rooms = [r.name for r in classrooms if r.capacity >= c.student_count]
        print(f"      - {c.name} ({c.teacher}, {c.student_count}人) "
              f"适用教室: {len(valid_rooms)}间")

    print("\n  [步骤3/7] 生成初始排课方案...")
    temp_schedule = Schedule(
        courses=expanded_courses,
        classrooms=classrooms,
        timeslots=timeslots,
        assignments=[]
    )
    if use_random_initial:
        _, initial_assignments = generate_random_schedule(schedule)
        print(f"    ✓ 已生成随机初始方案")
    else:
        initial_assignments = []
        np.random.seed(random_seed)
        for i, c in enumerate(expanded_courses):
            valid_rooms = [r.classroom_id for r in classrooms
                           if r.capacity >= c.student_count]
            ts_id = np.random.randint(0, len(timeslots))
            room_id = valid_rooms[i % len(valid_rooms)] if valid_rooms else 0
            initial_assignments.append(ScheduleAssignment(
                course_id=i, timeslot_id=ts_id, classroom_id=room_id
            ))
        print(f"    ✓ 已生成半随机初始方案 (种子: {random_seed})")

    if locked_courses:
        for lock in locked_courses:
            if lock.course_id < len(initial_assignments):
                a = initial_assignments[lock.course_id]
                if lock.lock_timeslot:
                    a.timeslot_id = lock.timeslot_id
                if lock.lock_classroom:
                    a.classroom_id = lock.classroom_id
        print(f"    ✓ 已应用 {len(locked_courses)} 个手动锁定")

    print("\n  [步骤4/7] 检测初始方案冲突...")
    detector = ConflictDetector(temp_schedule)
    report_before = detector.detect(initial_assignments)
    conflicting_before = detector.get_conflicting_courses(report_before)
    print_conflict_report(report_before, "初始方案冲突检测报告")

    print("    生成初始课表可视化...")
    visualize_timetable(
        schedule, expanded_courses, initial_assignments,
        conflicting_courses=conflicting_before,
        title="初始课表 (含冲突标记)",
        save_path=os.path.join(OUTPUT_DIR, "01_timetable_initial.png")
    )
    visualize_conflict_detailed(
        report_before,
        save_path=os.path.join(OUTPUT_DIR, "02_conflicts_initial.png")
    )

    print("\n  [步骤5/7] 遗传算法优化排课...")
    ga_kwargs = dict(
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        elite_count=elite_count,
        random_seed=random_seed,
        preferences=preferences,
        locked_courses=locked_courses,
    )
    if use_improved_ga:
        ga_kwargs.update(dict(
            fitness_sharing=fitness_sharing,
            adaptive_mutation=adaptive_mutation,
            diversity_preserve=diversity_preserve,
            migration_rate=migration_rate,
        ))
        mode_str = "增强模式 (矢量化+抗早熟)"
    else:
        ga_kwargs.update(dict(
            fitness_sharing=False,
            adaptive_mutation=False,
            diversity_preserve=False,
            migration_rate=0.0,
        ))
        mode_str = "基础模式"
    print(f"    模式: {mode_str}")
    print(f"    参数: 种群={population_size}, 最大代数={max_generations}, "
          f"初始变异率={mutation_rate}, 精英保留={elite_count}")

    ga = GeneticAlgorithmScheduler(
        schedule=schedule,
        expanded_courses=expanded_courses,
        **ga_kwargs
    )

    optimized_assignments, report_after, history, pref_result = ga.optimize(
        initial_assignments=initial_assignments,
        target_conflicts=0,
        patience=120,
        verbose=True,
    )
    conflicting_after = detector.get_conflicting_courses(report_after)

    print_conflict_report(report_after, "优化后方案冲突检测报告")
    print_preference_result(pref_result)

    print(f"\n  算法统计: 总代数={len(history.generations)}, "
          f"耗时={history.elapsed_time:.3f}秒, "
          f"重启次数={getattr(history, 'restarts', 0)}, "
          f"最终多样性={history.diversity[-1] if history.diversity else 'N/A'}")

    print("\n  [步骤6/7] 生成可视化结果与报告...")
    adjustments = find_adjustments(initial_assignments, optimized_assignments)
    adjusted_ids = {cid for cid, _, _ in adjustments}

    locked_ids = {l.course_id for l in locked_courses} if locked_courses else set()
    adjusted_ids -= locked_ids

    print_adjustments(adjustments, expanded_courses, schedule)

    visualize_timetable(
        schedule, expanded_courses, optimized_assignments,
        conflicting_courses=conflicting_after,
        adjusted_courses=adjusted_ids,
        locked_courses=locked_ids,
        title="优化后课表 (★=已调整, 🔒=已锁定, ✖=仍有冲突)",
        save_path=os.path.join(OUTPUT_DIR, "03_timetable_optimized.png")
    )
    visualize_convergence(
        history,
        save_path=os.path.join(OUTPUT_DIR, "04_convergence.png")
    )
    visualize_teacher_distribution(
        schedule, expanded_courses, optimized_assignments,
        save_path=os.path.join(OUTPUT_DIR, "05_teacher_distribution.png")
    )
    visualize_adjustments(
        schedule, expanded_courses,
        initial_assignments, optimized_assignments, adjustments,
        save_path=os.path.join(OUTPUT_DIR, "06_adjustment_report.png")
    )
    if report_after.conflicts:
        visualize_conflict_detailed(
            report_after,
            save_path=os.path.join(OUTPUT_DIR, "07_conflicts_final.png")
        )

    save_results_to_json(
        expanded_courses, initial_assignments, optimized_assignments,
        adjustments, report_before, report_after, history, pref_result,
        os.path.join(OUTPUT_DIR, "scheduling_result.json")
    )

    print_timetable_summary(expanded_courses, optimized_assignments, schedule)

    print("\n" + "="*60)
    print("  排课任务完成! 输出文件列表:")
    print("="*60)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname:<40s} {size_kb:7.1f} KB")
    print("="*60)

    return optimized_assignments, report_after, adjustments, history, pref_result


def main():
    parser = argparse.ArgumentParser(
        description="大学教务排课辅助系统 - 冲突检测与遗传算法优化"
    )
    parser.add_argument("--pop-size", type=int, default=80,
                        help="遗传算法种群大小 (默认: 80)")
    parser.add_argument("--max-gen", type=int, default=300,
                        help="最大进化代数 (默认: 300)")
    parser.add_argument("--mutation-rate", type=float, default=0.08,
                        help="初始变异概率 (默认: 0.08)")
    parser.add_argument("--elite-count", type=int, default=4,
                        help="精英保留数量 (默认: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--random-initial", action="store_true",
                        help="使用完全随机的初始方案")
    parser.add_argument("--output", type=str, default="output",
                        help="输出目录 (默认: output)")
    parser.add_argument("--basic-mode", action="store_true",
                        help="禁用抗早熟收敛机制(基础模式)，用于对比")
    parser.add_argument("--no-fitness-sharing", action="store_true",
                        help="禁用适应度共享")
    parser.add_argument("--no-adaptive-mutation", action="store_true",
                        help="禁用自适应变异率")
    parser.add_argument("--no-diversity", action="store_true",
                        help="禁用多样性保持与部分重启")
    parser.add_argument("--migration-rate", type=float, default=0.05,
                        help="随机移民比例 (默认: 0.05)")
    parser.add_argument("--no-preferences", action="store_true",
                        help="禁用偏好约束")
    parser.add_argument("--no-locks", action="store_true",
                        help="禁用手动锁定")

    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output

    try:
        run_scheduler(
            population_size=args.pop_size,
            max_generations=args.max_gen,
            mutation_rate=args.mutation_rate,
            elite_count=args.elite_count,
            random_seed=args.seed,
            use_random_initial=args.random_initial,
            use_improved_ga=not args.basic_mode,
            fitness_sharing=not args.no_fitness_sharing,
            adaptive_mutation=not args.no_adaptive_mutation,
            diversity_preserve=not args.no_diversity,
            migration_rate=args.migration_rate,
            use_preferences=not args.no_preferences,
            use_locks=not args.no_locks,
        )
    except KeyboardInterrupt:
        print("\n\n用户中断执行。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
