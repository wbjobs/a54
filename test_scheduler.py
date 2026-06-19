import sys
import os
import unittest
import numpy as np
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_structures import (
    Course, Classroom, TimeSlot, Schedule, ScheduleAssignment,
    create_sample_data, expand_courses_to_sessions, create_default_timeslots
)
from conflict_detector import (
    ConflictDetector, ConflictReport, generate_random_schedule
)
from genetic_scheduler import (
    GeneticAlgorithmScheduler, GAHistory, PreferenceResult, find_adjustments
)
from data_structures import Preference, LockedCourse, create_sample_preferences, create_sample_locks


class TestDataStructures(unittest.TestCase):

    def test_create_default_timeslots(self):
        slots = create_default_timeslots()
        self.assertEqual(len(slots), 25)
        self.assertEqual(slots[0].day, 0)
        self.assertEqual(slots[0].period, 0)
        self.assertEqual(slots[24].day, 4)
        self.assertEqual(slots[24].period, 4)
        self.assertEqual(slots[0].day_name, "周一")
        self.assertEqual(slots[0].period_name, "第1-2节")

    def test_course_creation(self):
        c = Course(0, "数学", "张老师", ["一班"], 4, student_count=40)
        self.assertEqual(c.course_id, 0)
        self.assertEqual(c.name, "数学")
        self.assertEqual(c.student_count, 40)

    def test_course_default_student_count(self):
        c = Course(1, "物理", "李老师", ["一班", "二班"], 2)
        self.assertEqual(c.student_count, 80)

    def test_expand_courses_to_sessions(self):
        courses = [
            Course(0, "高数", "张", ["计1"], 4),
            Course(1, "线代", "李", ["计1"], 2),
            Course(2, "英语", "王", ["计1"], 3),
        ]
        expanded = expand_courses_to_sessions(courses)
        self.assertEqual(len(expanded), 2 + 1 + 2)

        session_names = [c.name for c in expanded]
        self.assertIn("高数(1)", session_names)
        self.assertIn("高数(2)", session_names)
        self.assertIn("线代(1)", session_names)
        self.assertIn("英语(1)", session_names)
        self.assertIn("英语(2)", session_names)

    def test_schedule_maps(self):
        courses, classrooms, timeslots = create_sample_data()
        s = Schedule(courses=courses, classrooms=classrooms, timeslots=timeslots)
        self.assertEqual(s.num_courses(), len(courses))
        self.assertEqual(s.num_classrooms(), len(classrooms))
        self.assertEqual(s.num_timeslots(), len(timeslots))

        cm = s.get_course_map()
        self.assertEqual(cm[0].name, "高等数学")
        rm = s.get_classroom_map()
        self.assertIsNotNone(rm[0])
        tm = s.get_timeslot_map()
        self.assertEqual(tm[0].day, 0)


class TestConflictDetection(unittest.TestCase):

    def setUp(self):
        self.courses = [
            Course(0, "高数", "张教授", ["计1"], 2, student_count=40),
            Course(1, "线代", "张教授", ["计2"], 2, student_count=40),
            Course(2, "英语", "王教授", ["计1"], 2, student_count=40),
            Course(3, "物理", "李教授", ["计3"], 2, student_count=40),
        ]
        self.classrooms = [
            Classroom(0, "A101", 60, "普通"),
            Classroom(1, "A102", 60, "普通"),
        ]
        self.timeslots = create_default_timeslots()
        self.schedule = Schedule(
            courses=self.courses,
            classrooms=self.classrooms,
            timeslots=self.timeslots,
        )
        self.detector = ConflictDetector(self.schedule)

    def test_no_conflict(self):
        assignments = [
            ScheduleAssignment(0, 0, 0),
            ScheduleAssignment(1, 1, 0),
            ScheduleAssignment(2, 2, 1),
            ScheduleAssignment(3, 3, 0),
        ]
        report = self.detector.detect(assignments)
        self.assertEqual(report.total_conflicts(), 0)
        self.assertFalse(report.has_conflicts())

    def test_teacher_conflict(self):
        assignments = [
            ScheduleAssignment(0, 0, 0),
            ScheduleAssignment(1, 0, 1),
        ]
        report = self.detector.detect(assignments)
        self.assertEqual(report.teacher_conflicts, 1)

    def test_class_conflict(self):
        self.courses[1].teacher = "不同老师"
        self.courses[1].classes = ["计1"]
        assignments = [
            ScheduleAssignment(0, 0, 0),
            ScheduleAssignment(1, 0, 1),
        ]
        report = self.detector.detect(assignments)
        self.assertEqual(report.class_conflicts, 1)

    def test_classroom_conflict(self):
        self.courses[1].teacher = "不同老师"
        assignments = [
            ScheduleAssignment(0, 0, 0),
            ScheduleAssignment(1, 0, 0),
        ]
        report = self.detector.detect(assignments)
        self.assertEqual(report.classroom_conflicts, 1)

    def test_capacity_violation(self):
        self.courses[0].student_count = 100
        assignments = [ScheduleAssignment(0, 0, 0)]
        report = self.detector.detect(assignments)
        self.assertEqual(report.capacity_violations, 1)

    def test_multiple_conflicts(self):
        self.courses[1].classes = ["计1"]
        assignments = [
            ScheduleAssignment(0, 0, 0),
            ScheduleAssignment(1, 0, 0),
        ]
        report = self.detector.detect(assignments)
        self.assertGreater(report.teacher_conflicts, 0)
        self.assertGreater(report.class_conflicts, 0)
        self.assertGreater(report.classroom_conflicts, 0)
        self.assertGreater(report.total_conflicts(), 2)

    def test_get_conflicting_courses(self):
        assignments = [
            ScheduleAssignment(0, 0, 0),
            ScheduleAssignment(1, 0, 1),
            ScheduleAssignment(3, 5, 0),
        ]
        report = self.detector.detect(assignments)
        conflicting = self.detector.get_conflicting_courses(report)
        self.assertIn(0, conflicting)
        self.assertIn(1, conflicting)
        self.assertNotIn(3, conflicting)

    def test_generate_random_schedule(self):
        base_schedule = Schedule(
            courses=[
                Course(0, "高数", "张", ["计1"], 4),
                Course(1, "线代", "李", ["计1"], 2),
            ],
            classrooms=self.classrooms,
            timeslots=self.timeslots,
        )
        expanded, assignments = generate_random_schedule(base_schedule)
        self.assertEqual(len(assignments), 3)
        for a in assignments:
            self.assertLess(a.timeslot_id, len(self.timeslots))
            self.assertLess(a.classroom_id, len(self.classrooms))


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.base_courses = [
            Course(0, "高数", "张教授", ["计1"], 4),
            Course(1, "线代", "李教授", ["计1"], 2),
            Course(2, "英语", "王教授", ["计2"], 2),
            Course(3, "数据结构", "赵教授", ["计1", "计2"], 4),
        ]
        self.classrooms = [
            Classroom(0, "A101", 120, "普通"),
            Classroom(1, "A102", 120, "普通"),
            Classroom(2, "A103", 60, "普通"),
        ]
        self.timeslots = create_default_timeslots()
        self.schedule = Schedule(
            courses=self.base_courses,
            classrooms=self.classrooms,
            timeslots=self.timeslots,
        )
        self.expanded = expand_courses_to_sessions(self.base_courses)

    def test_valid_rooms_computation(self):
        ga = GeneticAlgorithmScheduler(
            schedule=self.schedule,
            expanded_courses=self.expanded,
            population_size=10,
            max_generations=1,
            random_seed=42,
        )
        self.assertEqual(len(ga.valid_rooms_per_course), len(self.expanded))
        for cid, valid in ga.valid_rooms_per_course.items():
            self.assertGreater(len(valid), 0)

    def test_encode_decode_consistency(self):
        ga = GeneticAlgorithmScheduler(
            schedule=self.schedule,
            expanded_courses=self.expanded,
            population_size=10,
            max_generations=1,
            random_seed=42,
        )
        assignments = [
            ScheduleAssignment(i, i % 5, i % 3)
            for i in range(len(self.expanded))
        ]
        chrom = ga._encode_chromosome(assignments)
        decoded = ga._decode_chromosome(chrom)
        self.assertEqual(len(decoded), len(assignments))
        for i, (orig, dec) in enumerate(zip(assignments, decoded)):
            self.assertEqual(orig.timeslot_id, dec.timeslot_id)
            self.assertEqual(orig.classroom_id, dec.classroom_id)

    def test_fitness_evaluation(self):
        ga = GeneticAlgorithmScheduler(
            schedule=self.schedule,
            expanded_courses=self.expanded,
            population_size=10,
            max_generations=1,
            random_seed=42,
        )
        chrom = np.zeros((len(self.expanded), 2), dtype=np.int32)
        for i in range(len(self.expanded)):
            chrom[i, 0] = i % 10
            chrom[i, 1] = 0
        fitness, report = ga._evaluate_fitness(chrom)
        self.assertGreater(fitness, 0)
        self.assertLessEqual(fitness, 1)

    def test_population_initialization(self):
        ga = GeneticAlgorithmScheduler(
            schedule=self.schedule,
            expanded_courses=self.expanded,
            population_size=20,
            max_generations=1,
            random_seed=42,
        )
        init = [ScheduleAssignment(i, 0, 0) for i in range(len(self.expanded))]
        pop = ga._initialize_population(init)
        self.assertEqual(pop.shape, (20, len(self.expanded), 2))
        self.assertTrue(np.array_equal(pop[0], ga._encode_chromosome(init)))

    def test_crossover_operation(self):
        ga = GeneticAlgorithmScheduler(
            schedule=self.schedule,
            expanded_courses=self.expanded,
            population_size=10,
            max_generations=1,
            crossover_rate=1.0,
            random_seed=42,
        )
        p1 = np.zeros((len(self.expanded), 2), dtype=np.int32)
        p2 = np.ones((len(self.expanded), 2), dtype=np.int32)
        np.random.seed(0)
        c1, c2 = ga._crossover(p1, p2)
        self.assertEqual(c1.shape, p1.shape)
        self.assertEqual(c2.shape, p2.shape)
        combined_diff = np.sum(c1 != p1) + np.sum(c2 != p2)
        self.assertGreater(combined_diff, 0)

    def test_mutation_operation(self):
        ga = GeneticAlgorithmScheduler(
            schedule=self.schedule,
            expanded_courses=self.expanded,
            population_size=10,
            max_generations=1,
            mutation_rate=1.0,
            random_seed=42,
        )
        chrom = np.zeros((len(self.expanded), 2), dtype=np.int32)
        fake_report = ConflictReport([], 0, 0, 0, 0, 0)
        mutated = ga._mutate(chrom, fake_report)
        diff_count = np.sum(chrom != mutated)
        self.assertGreater(diff_count, 0)

    def test_short_optimization_run(self):
        ga = GeneticAlgorithmScheduler(
            schedule=self.schedule,
            expanded_courses=self.expanded,
            population_size=20,
            max_generations=20,
            mutation_rate=0.1,
            elite_count=2,
            random_seed=42,
        )
        initial = [
            ScheduleAssignment(i, 0, 0)
            for i in range(len(self.expanded))
        ]
        optimized, report, history, pref_result = ga.optimize(
            initial_assignments=initial,
            target_conflicts=0,
            patience=10,
            verbose=False,
        )
        self.assertEqual(len(optimized), len(self.expanded))
        self.assertGreater(len(history.generations), 0)
        self.assertEqual(len(history.best_fitness), len(history.generations))
        self.assertEqual(len(history.conflict_counts), len(history.generations))
        self.assertGreater(history.elapsed_time, 0)
        self.assertIsNotNone(pref_result)

    def test_find_adjustments(self):
        old = [
            ScheduleAssignment(0, 0, 0),
            ScheduleAssignment(1, 1, 1),
            ScheduleAssignment(2, 2, 2),
        ]
        new = [
            ScheduleAssignment(0, 0, 0),
            ScheduleAssignment(1, 5, 1),
            ScheduleAssignment(2, 2, 3),
        ]
        adjustments = find_adjustments(old, new)
        self.assertEqual(len(adjustments), 2)
        ids = [a[0] for a in adjustments]
        self.assertIn(1, ids)
        self.assertIn(2, ids)


class TestEndToEnd(unittest.TestCase):

    def test_full_sample_pipeline(self):
        courses, classrooms, timeslots = create_sample_data()
        base_schedule = Schedule(
            courses=courses,
            classrooms=classrooms,
            timeslots=timeslots,
        )
        expanded = expand_courses_to_sessions(courses)
        temp_schedule = Schedule(
            courses=expanded,
            classrooms=classrooms,
            timeslots=timeslots,
        )

        np.random.seed(123)
        initial = []
        for i, c in enumerate(expanded):
            valid_rooms = [r.classroom_id for r in classrooms
                           if r.capacity >= c.student_count]
            initial.append(ScheduleAssignment(
                course_id=i,
                timeslot_id=np.random.randint(0, len(timeslots)),
                classroom_id=valid_rooms[i % len(valid_rooms)] if valid_rooms else 0
            ))

        detector = ConflictDetector(temp_schedule)
        report_before = detector.detect(initial)
        self.assertIsNotNone(report_before)

        ga = GeneticAlgorithmScheduler(
            schedule=base_schedule,
            expanded_courses=expanded,
            population_size=30,
            max_generations=50,
            mutation_rate=0.08,
            elite_count=3,
            random_seed=42,
        )
        optimized, report_after, history, pref_result = ga.optimize(
            initial_assignments=initial,
            target_conflicts=0,
            patience=20,
            verbose=False,
        )

        self.assertEqual(len(optimized), len(expanded))
        self.assertIsNotNone(report_after)
        self.assertIsNotNone(history)
        self.assertGreater(len(history.generations), 0)
        self.assertIsNotNone(pref_result)

        if report_before.has_conflicts():
            adjustments = find_adjustments(initial, optimized)
            self.assertIsInstance(adjustments, list)


class TestPreferences(unittest.TestCase):

    def test_preference_creation(self):
        pref = Preference(
            pref_id=0,
            pref_type="teacher_time",
            target="张教授",
            target_name="张教授",
            allowed_days=[1],
            allowed_periods=[0, 1],
            priority="hard",
            weight=8.0,
            description="张教授只愿意在周二上午上课"
        )
        self.assertTrue(pref.is_satisfied(1, 0, 0))
        self.assertFalse(pref.is_satisfied(0, 0, 0))
        self.assertFalse(pref.is_satisfied(1, 3, 0))
        self.assertEqual(pref.priority, "hard")

    def test_preference_classroom_constraint(self):
        pref = Preference(
            pref_id=1,
            pref_type="course_room",
            target="人工智能",
            target_name="人工智能",
            allowed_classroom_ids=[6, 7],
            priority="soft",
            weight=3.0,
        )
        self.assertTrue(pref.is_satisfied(0, 0, 6))
        self.assertFalse(pref.is_satisfied(0, 0, 0))

    def test_create_sample_preferences(self):
        prefs = create_sample_preferences()
        self.assertEqual(len(prefs), 5)
        hard_count = sum(1 for p in prefs if p.priority == "hard")
        self.assertEqual(hard_count, 2)

    def test_ga_with_preferences(self):
        courses, classrooms, timeslots = create_sample_data()
        schedule = Schedule(
            courses=courses, classrooms=classrooms,
            timeslots=timeslots, assignments=[]
        )
        expanded = expand_courses_to_sessions(courses)
        prefs = create_sample_preferences()

        assignments = []
        np.random.seed(123)
        for i, c in enumerate(expanded):
            valid_rooms = [r.classroom_id for r in classrooms
                           if r.capacity >= c.student_count]
            ts_id = np.random.randint(0, len(timeslots))
            room_id = valid_rooms[i % len(valid_rooms)] if valid_rooms else 0
            assignments.append(ScheduleAssignment(
                course_id=i, timeslot_id=ts_id, classroom_id=room_id
            ))

        ga = GeneticAlgorithmScheduler(
            schedule=schedule,
            expanded_courses=expanded,
            population_size=60,
            max_generations=150,
            random_seed=123,
            preferences=prefs,
        )
        self.assertTrue(ga.has_preferences)
        self.assertGreater(ga.total_pref_weight, 0)

        optimized, report, history, pref_result = ga.optimize(
            initial_assignments=assignments,
            target_conflicts=0,
            patience=80,
            verbose=False,
        )

        self.assertEqual(report.total_conflicts(), 0)
        self.assertEqual(pref_result.total_preferences, 5)
        self.assertGreaterEqual(pref_result.satisfied_count, 0)
        self.assertLessEqual(pref_result.satisfied_count, 5)
        self.assertGreaterEqual(pref_result.satisfaction_rate, 0.0)
        self.assertLessEqual(pref_result.satisfaction_rate, 1.0)


class TestLockedCourses(unittest.TestCase):

    def test_locked_course_creation(self):
        lock = LockedCourse(
            course_id=3,
            timeslot_id=5,
            classroom_id=2,
            lock_timeslot=True,
            lock_classroom=True,
            reason="实验设备已预约"
        )
        self.assertEqual(lock.course_id, 3)
        self.assertEqual(lock.timeslot_id, 5)
        self.assertEqual(lock.classroom_id, 2)
        self.assertTrue(lock.lock_timeslot)
        self.assertTrue(lock.lock_classroom)

    def test_create_sample_locks(self):
        locks = create_sample_locks()
        self.assertGreater(len(locks), 0)

    def test_locks_preserved_in_initial_population(self):
        courses, classrooms, timeslots = create_sample_data()
        schedule = Schedule(
            courses=courses, classrooms=classrooms,
            timeslots=timeslots, assignments=[]
        )
        expanded = expand_courses_to_sessions(courses)

        lock_cid = 8
        lock_ts = 12
        lock_rm = 4
        locks = [LockedCourse(
            course_id=lock_cid,
            timeslot_id=lock_ts,
            classroom_id=lock_rm,
            lock_timeslot=True,
            lock_classroom=True,
            reason="测试锁定"
        )]

        ga = GeneticAlgorithmScheduler(
            schedule=schedule,
            expanded_courses=expanded,
            population_size=30,
            max_generations=10,
            random_seed=42,
            locked_courses=locks,
        )

        self.assertEqual(ga.num_locked, 1)
        self.assertTrue(ga.locked_ts_mask[lock_cid])
        self.assertTrue(ga.locked_rm_mask[lock_cid])

        pop = ga._initialize_population([])
        for i in range(pop.shape[0]):
            self.assertEqual(pop[i, lock_cid, 0], lock_ts)
            self.assertEqual(pop[i, lock_cid, 1], lock_rm)

    def test_locks_preserved_after_optimization(self):
        courses, classrooms, timeslots = create_sample_data()
        schedule = Schedule(
            courses=courses, classrooms=classrooms,
            timeslots=timeslots, assignments=[]
        )
        expanded = expand_courses_to_sessions(courses)

        lock_cid = 8
        lock_ts = 12
        lock_rm = 4
        locks = [LockedCourse(
            course_id=lock_cid,
            timeslot_id=lock_ts,
            classroom_id=lock_rm,
            lock_timeslot=True,
            lock_classroom=True,
            reason="测试锁定"
        )]

        assignments = []
        np.random.seed(77)
        for i, c in enumerate(expanded):
            valid_rooms = [r.classroom_id for r in classrooms
                           if r.capacity >= c.student_count]
            ts_id = np.random.randint(0, len(timeslots))
            room_id = valid_rooms[i % len(valid_rooms)] if valid_rooms else 0
            assignments.append(ScheduleAssignment(
                course_id=i, timeslot_id=ts_id, classroom_id=room_id
            ))

        ga = GeneticAlgorithmScheduler(
            schedule=schedule,
            expanded_courses=expanded,
            population_size=50,
            max_generations=100,
            random_seed=77,
            locked_courses=locks,
        )

        optimized, report, history, pref_result = ga.optimize(
            initial_assignments=assignments,
            target_conflicts=0,
            patience=50,
            verbose=False,
        )

        locked_a = None
        for a in optimized:
            if a.course_id == lock_cid:
                locked_a = a
                break
        self.assertIsNotNone(locked_a)
        self.assertEqual(locked_a.timeslot_id, lock_ts)
        self.assertEqual(locked_a.classroom_id, lock_rm)


def run_tests():
    print("=" * 60)
    print("  运行排课系统单元测试")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDataStructures))
    suite.addTests(loader.loadTestsFromTestCase(TestConflictDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestGeneticAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestPreferences))
    suite.addTests(loader.loadTestsFromTestCase(TestLockedCourses))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("  [OK] 所有测试通过!")
    else:
        print(f"  [FAIL] 有 {len(result.failures)} 个失败, "
              f"{len(result.errors)} 个错误")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
