import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from data_structures import (
    Course, Classroom, TimeSlot, Schedule, ScheduleAssignment,
    expand_courses_to_sessions
)


@dataclass
class Conflict:
    conflict_type: str
    course_id_1: int
    course_id_2: int
    timeslot_id: int
    detail: str = ""


@dataclass
class ConflictReport:
    conflicts: List[Conflict]
    teacher_conflicts: int
    class_conflicts: int
    classroom_conflicts: int
    capacity_violations: int
    type_violations: int

    def total_conflicts(self) -> int:
        return (self.teacher_conflicts + self.class_conflicts +
                self.classroom_conflicts + self.capacity_violations + self.type_violations)

    def has_conflicts(self) -> bool:
        return self.total_conflicts() > 0


@dataclass
class Adjustment:
    course_id: int
    old_timeslot_id: int
    new_timeslot_id: int
    old_classroom_id: int
    new_classroom_id: int


class ConflictDetector:
    def __init__(self, schedule: Schedule):
        self.schedule = schedule
        self.course_map = schedule.get_course_map()
        self.classroom_map = schedule.get_classroom_map()
        self.timeslot_map = schedule.get_timeslot_map()

    def detect(self, assignments: List[ScheduleAssignment] = None) -> ConflictReport:
        if assignments is None:
            assignments = self.schedule.assignments

        conflicts: List[Conflict] = []
        teacher_conflicts = 0
        class_conflicts = 0
        classroom_conflicts = 0
        capacity_violations = 0
        type_violations = 0

        ts_groups: Dict[int, List[ScheduleAssignment]] = {}
        for a in assignments:
            if a.timeslot_id not in ts_groups:
                ts_groups[a.timeslot_id] = []
            ts_groups[a.timeslot_id].append(a)

        for ts_id, ts_assigns in ts_groups.items():
            n = len(ts_assigns)
            for i in range(n):
                a1 = ts_assigns[i]
                c1 = self.course_map[a1.course_id]
                r1 = self.classroom_map[a1.classroom_id]

                if c1.student_count > r1.capacity:
                    capacity_violations += 1
                    conflicts.append(Conflict(
                        conflict_type="capacity",
                        course_id_1=a1.course_id,
                        course_id_2=-1,
                        timeslot_id=ts_id,
                        detail=f"课程{c1.name}学生数{c1.student_count} > 教室{r1.name}容量{r1.capacity}"
                    ))

                for j in range(i + 1, n):
                    a2 = ts_assigns[j]
                    c2 = self.course_map[a2.course_id]
                    r2 = self.classroom_map[a2.classroom_id]

                    if c1.teacher == c2.teacher:
                        teacher_conflicts += 1
                        conflicts.append(Conflict(
                            conflict_type="teacher",
                            course_id_1=a1.course_id,
                            course_id_2=a2.course_id,
                            timeslot_id=ts_id,
                            detail=f"教师{c1.teacher}同时上{c1.name}和{c2.name}"
                        ))

                    classes1 = set(c1.classes)
                    classes2 = set(c2.classes)
                    if classes1 & classes2:
                        class_conflicts += 1
                        conflicts.append(Conflict(
                            conflict_type="class",
                            course_id_1=a1.course_id,
                            course_id_2=a2.course_id,
                            timeslot_id=ts_id,
                            detail=f"班级{classes1 & classes2}同时上{c1.name}和{c2.name}"
                        ))

                    if a1.classroom_id == a2.classroom_id:
                        classroom_conflicts += 1
                        conflicts.append(Conflict(
                            conflict_type="classroom",
                            course_id_1=a1.course_id,
                            course_id_2=a2.course_id,
                            timeslot_id=ts_id,
                            detail=f"教室{r1.name}同时被{c1.name}和{c2.name}占用"
                        ))

        return ConflictReport(
            conflicts=conflicts,
            teacher_conflicts=teacher_conflicts,
            class_conflicts=class_conflicts,
            classroom_conflicts=classroom_conflicts,
            capacity_violations=capacity_violations,
            type_violations=type_violations,
        )

    def detect_conflict_matrix(self, assignments: List[ScheduleAssignment]) -> np.ndarray:
        n = len(assignments)
        matrix = np.zeros((n, n), dtype=np.int32)
        conflict_map = {a.course_id: idx for idx, a in enumerate(assignments)}

        for i, a1 in enumerate(assignments):
            c1 = self.course_map[a1.course_id]
            r1 = self.classroom_map[a1.classroom_id]

            if c1.student_count > r1.capacity:
                matrix[i, i] += 1

            for j, a2 in enumerate(assignments):
                if i >= j:
                    continue
                if a1.timeslot_id != a2.timeslot_id:
                    continue

                c2 = self.course_map[a2.course_id]

                if c1.teacher == c2.teacher:
                    matrix[i, j] += 1
                    matrix[j, i] += 1

                if set(c1.classes) & set(c2.classes):
                    matrix[i, j] += 1
                    matrix[j, i] += 1

                if a1.classroom_id == a2.classroom_id:
                    matrix[i, j] += 1
                    matrix[j, i] += 1

        return matrix

    def get_conflicting_courses(self, report: ConflictReport) -> Set[int]:
        conflicting = set()
        for c in report.conflicts:
            conflicting.add(c.course_id_1)
            if c.course_id_2 >= 0:
                conflicting.add(c.course_id_2)
        return conflicting


def generate_random_schedule(schedule: Schedule) -> List[ScheduleAssignment]:
    expanded_courses = expand_courses_to_sessions(schedule.courses)
    num_courses = len(expanded_courses)
    num_timeslots = schedule.num_timeslots()
    num_classrooms = schedule.num_classrooms()

    assignments = []
    temp_schedule = Schedule(
        courses=expanded_courses,
        classrooms=schedule.classrooms,
        timeslots=schedule.timeslots,
        assignments=[]
    )
    temp_schedule_map = temp_schedule.get_course_map()
    classroom_map = schedule.get_classroom_map()

    for i in range(num_courses):
        course = temp_schedule_map[i]
        valid_rooms = [
            r.classroom_id for r in schedule.classrooms
            if r.capacity >= course.student_count
        ]
        if not valid_rooms:
            valid_rooms = [r.classroom_id for r in schedule.classrooms]

        ts_id = np.random.randint(0, num_timeslots)
        room_id = valid_rooms[np.random.randint(0, len(valid_rooms))]

        assignments.append(ScheduleAssignment(
            course_id=i,
            timeslot_id=ts_id,
            classroom_id=room_id
        ))

    return expanded_courses, assignments
