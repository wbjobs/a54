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


@dataclass
class Preference:
    pref_id: int
    pref_type: str
    target: str
    target_name: str
    allowed_days: List[int] = field(default_factory=list)
    allowed_periods: List[int] = field(default_factory=list)
    allowed_classroom_ids: List[int] = field(default_factory=list)
    priority: str = "hard"
    weight: float = 5.0
    description: str = ""

    def is_satisfied(self, day: int, period: int, classroom_id: int) -> bool:
        if self.allowed_days and day not in self.allowed_days:
            return False
        if self.allowed_periods and period not in self.allowed_periods:
            return False
        if self.allowed_classroom_ids and classroom_id not in self.allowed_classroom_ids:
            return False
        return True


@dataclass
class LockedCourse:
    course_id: int
    timeslot_id: int
    classroom_id: int
    lock_timeslot: bool = True
    lock_classroom: bool = True
    reason: str = ""


def create_sample_preferences() -> List[Preference]:
    prefs = []
    prefs.append(Preference(
        pref_id=0,
        pref_type="teacher_time",
        target="张教授",
        target_name="张教授",
        allowed_days=[1],
        allowed_periods=[0, 1],
        priority="hard",
        weight=8.0,
        description="张教授只愿意在周二上午上课"
    ))
    prefs.append(Preference(
        pref_id=1,
        pref_type="course_time",
        target="高等数学",
        target_name="高等数学",
        allowed_periods=[0, 1],
        priority="hard",
        weight=6.0,
        description="高等数学必须安排在上午"
    ))
    prefs.append(Preference(
        pref_id=2,
        pref_type="course_room",
        target="人工智能",
        target_name="人工智能",
        allowed_classroom_ids=[6, 7],
        priority="soft",
        weight=3.0,
        description="人工智能优先安排在多媒体教室"
    ))
    prefs.append(Preference(
        pref_id=3,
        pref_type="course_time",
        target="数据结构",
        target_name="数据结构",
        allowed_periods=[0, 1, 2],
        priority="soft",
        weight=2.0,
        description="数据结构尽量安排在白天（前6节）"
    ))
    prefs.append(Preference(
        pref_id=4,
        pref_type="teacher_time",
        target="李教授",
        target_name="李教授",
        allowed_days=[0, 2, 4],
        priority="soft",
        weight=2.5,
        description="李教授偏好周一、周三、周五上课"
    ))
    return prefs


def create_sample_locks() -> List[LockedCourse]:
    locks = []
    locks.append(LockedCourse(
        course_id=8,
        timeslot_id=12,
        classroom_id=4,
        lock_timeslot=True,
        lock_classroom=True,
        reason="计算机网络实验课，实验设备已预约"
    ))
    return locks
