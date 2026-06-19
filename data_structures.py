import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Course:
    course_id: int
    name: str
    teacher: str
    classes: List[str]
    weekly_hours: int
    student_count: int = 0

    def __post_init__(self):
        if self.student_count == 0:
            self.student_count = len(self.classes) * 40


@dataclass
class Classroom:
    classroom_id: int
    name: str
    capacity: int
    room_type: str


@dataclass
class TimeSlot:
    timeslot_id: int
    day: int
    period: int
    day_name: str = ""
    period_name: str = ""

    def __post_init__(self):
        day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        period_names = ["第1-2节", "第3-4节", "第5-6节", "第7-8节", "第9-10节"]
        if not self.day_name:
            self.day_name = day_names[self.day]
        if not self.period_name:
            self.period_name = period_names[self.period]


@dataclass
class ScheduleAssignment:
    course_id: int
    timeslot_id: int
    classroom_id: int


@dataclass
class Schedule:
    courses: List[Course]
    classrooms: List[Classroom]
    timeslots: List[TimeSlot]
    assignments: List[ScheduleAssignment] = field(default_factory=list)

    def get_course_map(self) -> Dict[int, Course]:
        return {c.course_id: c for c in self.courses}

    def get_classroom_map(self) -> Dict[int, Classroom]:
        return {r.classroom_id: r for r in self.classrooms}

    def get_timeslot_map(self) -> Dict[int, TimeSlot]:
        return {t.timeslot_id: t for t in self.timeslots}

    def num_courses(self) -> int:
        return len(self.courses)

    def num_classrooms(self) -> int:
        return len(self.classrooms)

    def num_timeslots(self) -> int:
        return len(self.timeslots)


def create_default_timeslots() -> List[TimeSlot]:
    timeslots = []
    tid = 0
    for day in range(5):
        for period in range(5):
            timeslots.append(TimeSlot(timeslot_id=tid, day=day, period=period))
            tid += 1
    return timeslots


def create_sample_data() -> Tuple[List[Course], List[Classroom], List[TimeSlot]]:
    courses = [
        Course(0, "高等数学", "张教授", ["计科1班", "计科2班"], 4),
        Course(1, "线性代数", "李教授", ["计科1班"], 2),
        Course(2, "离散数学", "王教授", ["计科2班"], 2),
        Course(3, "数据结构", "赵教授", ["计科1班", "计科2班"], 4),
        Course(4, "操作系统", "刘教授", ["计科1班"], 3),
        Course(5, "计算机网络", "陈教授", ["计科2班"], 3),
        Course(6, "数据库原理", "周教授", ["计科1班", "计科2班"], 3),
        Course(7, "软件工程", "吴教授", ["计科1班"], 2),
        Course(8, "编译原理", "郑教授", ["计科2班"], 2),
        Course(9, "人工智能", "孙教授", ["计科1班", "计科2班"], 2),
        Course(10, "大学英语", "钱教授", ["计科1班", "计科2班"], 2),
        Course(11, "思想政治", "马教授", ["计科1班", "计科2班"], 2),
    ]

    classrooms = [
        Classroom(0, "教学楼A-101", 120, "普通教室"),
        Classroom(1, "教学楼A-102", 120, "普通教室"),
        Classroom(2, "教学楼A-103", 60, "普通教室"),
        Classroom(3, "教学楼A-104", 60, "普通教室"),
        Classroom(4, "实验楼B-201", 50, "实验室"),
        Classroom(5, "实验楼B-202", 50, "实验室"),
        Classroom(6, "教学楼A-201", 100, "多媒体教室"),
        Classroom(7, "教学楼A-202", 100, "多媒体教室"),
    ]

    timeslots = create_default_timeslots()

    return courses, classrooms, timeslots


def expand_courses_to_sessions(courses: List[Course]) -> List[Course]:
    expanded = []
    for course in courses:
        sessions_needed = (course.weekly_hours + 1) // 2
        for i in range(sessions_needed):
            new_course = Course(
                course_id=len(expanded),
                name=f"{course.name}({i+1})",
                teacher=course.teacher,
                classes=course.classes.copy(),
                weekly_hours=2,
                student_count=course.student_count,
            )
            new_course.base_course_id = course.course_id
            expanded.append(new_course)
    return expanded
