import sys
import os
import io
import time
import numpy as np
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_structures import (
    Course, Classroom, TimeSlot, Schedule, ScheduleAssignment,
    expand_courses_to_sessions, create_default_timeslots
)
from genetic_scheduler import GeneticAlgorithmScheduler


def generate_large_dataset(
    n_courses_base: int = 100,
    seed: int = 42,
):
    np.random.seed(seed)

    last_names = ["张", "李", "王", "赵", "刘", "陈", "杨", "黄", "周", "吴",
                  "徐", "孙", "马", "朱", "胡", "郭", "何", "林", "罗", "高"]
    subjects = ["高等数学", "线性代数", "离散数学", "概率论", "数据结构",
                "操作系统", "计算机网络", "数据库", "软件工程", "编译原理",
                "人工智能", "机器学习", "计算机图形学", "算法设计", "大学英语",
                "思想政治", "大学物理", "程序设计", "信息安全", "云计算"]
    class_prefixes = ["计科", "软工", "网安", "人工智能", "大数据", "物联网"]
    class_suffixes = [f"{i+1:02d}班" for i in range(8)]

    all_classes = [f"{p}{s}" for p in class_prefixes for s in class_suffixes]
    teacher_names = [f"{ln}教授" for ln in last_names]

    courses = []
    for i in range(n_courses_base):
        teacher = teacher_names[i % len(teacher_names)]
        n_classes = np.random.choice([1, 2, 2, 3])
        start_cls = np.random.randint(0, len(all_classes))
        classes = []
        for j in range(n_classes):
            classes.append(all_classes[(start_cls + j) % len(all_classes)])
        hours = np.random.choice([2, 3, 4, 4])
        subject = subjects[i % len(subjects)]
        courses.append(Course(
            course_id=i,
            name=f"{subject}{i//len(subjects)+1}" if i >= len(subjects) else subject,
            teacher=teacher,
            classes=classes,
            weekly_hours=hours,
            student_count=n_classes * 35 + np.random.randint(0, 20),
        ))

    classrooms = []
    room_types = ["普通教室", "多媒体教室", "实验室", "机房"]
    for i in range(min(n_courses_base // 2, 40)):
        cap = np.random.choice([40, 60, 80, 100, 120, 150, 200])
        rt = room_types[np.random.randint(0, len(room_types))]
        classrooms.append(Classroom(
            classroom_id=i,
            name=f"{'A' if i%2==0 else 'B'}-{100+i:03d}",
            capacity=cap,
            room_type=rt,
        ))

    timeslots = create_default_timeslots()
    return courses, classrooms, timeslots


def run_benchmark(
    n_courses_base: int,
    mode: str = "improved",
    n_runs: int = 1,
    seed: int = 42,
):
    print(f"\n{'='*70}")
    print(f"  基准测试: {mode.upper()} 模式 | {n_courses_base}门基础课 | {n_runs}次运行")
    print(f"{'='*70}")

    all_results = []
    for run in range(n_runs):
        np.random.seed(seed + run * 1000)
        courses, classrooms, timeslots = generate_large_dataset(
            n_courses_base, seed=seed + run
        )
        schedule = Schedule(courses=courses, classrooms=classrooms,
                            timeslots=timeslots)
        expanded = expand_courses_to_sessions(courses)

        initial_assignments = []
        for i, c in enumerate(expanded):
            valid_rooms = [r.classroom_id for r in classrooms
                           if r.capacity >= c.student_count]
            if not valid_rooms:
                valid_rooms = [r.classroom_id for r in classrooms]
            ts_id = np.random.randint(0, len(timeslots))
            room_id = valid_rooms[np.random.randint(0, len(valid_rooms))]
            initial_assignments.append(ScheduleAssignment(
                course_id=i, timeslot_id=ts_id, classroom_id=room_id
            ))

        improved = (mode == "improved")
        ga = GeneticAlgorithmScheduler(
            schedule=schedule,
            expanded_courses=expanded,
            population_size=100,
            max_generations=500,
            mutation_rate=0.07,
            elite_count=5,
            random_seed=seed + run,
            fitness_sharing=improved,
            adaptive_mutation=improved,
            diversity_preserve=improved,
            migration_rate=0.05 if improved else 0.0,
        )

        t0 = time.time()
        optimized, report, history = ga.optimize(
            initial_assignments=initial_assignments,
            target_conflicts=0,
            patience=120,
            verbose=False,
        )
        elapsed = time.time() - t0

        result = {
            "run": run + 1,
            "total_courses": len(expanded),
            "generations": len(history.generations),
            "elapsed_sec": round(elapsed, 3),
            "conflicts_final": report.total_conflicts(),
            "fitness_final": round(history.best_fitness[-1], 6),
            "avg_gen_time_ms": round(elapsed / max(len(history.generations), 1) * 1000, 2),
            "restarts": getattr(history, 'restarts', 0),
            "diversity_final": round(history.diversity[-1], 4) if history.diversity else None,
            "converged": report.total_conflicts() == 0,
        }
        all_results.append(result)
        print(f"  Run {run+1}: 代数={result['generations']:4d} | "
              f"耗时={result['elapsed_sec']:6.2f}s | "
              f"每代={result['avg_gen_time_ms']:6.2f}ms | "
              f"冲突={result['conflicts_final']:3d} | "
              f"适应度={result['fitness_final']:.6f} | "
              f"多样性={result['diversity_final']} | "
              f"重启={result['restarts']:2d} | "
              f"{'✓ 收敛' if result['converged'] else '✗ 未收敛'}")

    if len(all_results) > 1:
        print(f"\n  --- 平均值 ---")
        def avg(k): return round(np.mean([r[k] for r in all_results]), 3)
        conv_rate = sum(r['converged'] for r in all_results) / len(all_results) * 100
        print(f"    平均代数:   {avg('generations'):.0f}")
        print(f"    平均耗时:   {avg('elapsed_sec'):.3f}s")
        print(f"    每代均值:   {avg('avg_gen_time_ms'):.2f}ms")
        print(f"    收敛率:     {conv_rate:.0f}%")
        print(f"    最终冲突:   {avg('conflicts_final'):.1f}")
        print(f"    平均重启:   {avg('restarts'):.1f}次")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="排课系统性能对比基准测试")
    parser.add_argument("--courses", type=int, default=50,
                        help="基础课程数(展开后约1.5倍)")
    parser.add_argument("--runs", type=int, default=3, help="每个模式运行次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--compare", action="store_true",
                        help="同时运行增强模式和基础模式进行对比")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║       大学教务排课系统 - 遗传算法性能与抗早熟收敛基准测试           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if args.compare:
        print("\n>>> 基础模式 (所有抗早熟机制关闭)")
        basic_results = run_benchmark(
            args.courses, "basic", n_runs=args.runs, seed=args.seed
        )
        print("\n>>> 增强模式 (矢量化+抗早熟全开)")
        imp_results = run_benchmark(
            args.courses, "improved", n_runs=args.runs, seed=args.seed
        )
        print(f"\n{'='*70}")
        print("  对比总结")
        print(f"{'='*70}")
        basic_conv = sum(r['converged'] for r in basic_results) / args.runs * 100
        imp_conv = sum(r['converged'] for r in imp_results) / args.runs * 100
        basic_time = np.mean([r['elapsed_sec'] for r in basic_results])
        imp_time = np.mean([r['elapsed_sec'] for r in imp_results])
        print(f"    基础模式收敛率: {basic_conv:5.1f}%  |  增强模式收敛率: {imp_conv:5.1f}%")
        print(f"    基础模式均耗时: {basic_time:5.2f}s  |  增强模式均耗时: {imp_time:5.2f}s")
        if basic_time > 0:
            print(f"    速度提升: {(basic_time/imp_time):.2f}x")
        print(f"    收敛率提升: +{(imp_conv-basic_conv):.1f}个百分点")
    else:
        imp_results = run_benchmark(
            args.courses, "improved", n_runs=args.runs, seed=args.seed
        )

    print("\n[基准测试完成]")


if __name__ == "__main__":
    main()
