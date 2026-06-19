import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import time

from data_structures import (
    Course, Classroom, TimeSlot, Schedule, ScheduleAssignment,
    expand_courses_to_sessions
)
from conflict_detector import (
    ConflictDetector, ConflictReport, Conflict, generate_random_schedule
)


@dataclass
class GAHistory:
    generations: List[int] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    avg_fitness: List[float] = field(default_factory=list)
    worst_fitness: List[float] = field(default_factory=list)
    conflict_counts: List[int] = field(default_factory=list)
    elapsed_time: float = 0.0


class GeneticAlgorithmScheduler:
    def __init__(
        self,
        schedule: Schedule,
        expanded_courses: List[Course],
        population_size: int = 100,
        max_generations: int = 500,
        mutation_rate: float = 0.08,
        crossover_rate: float = 0.85,
        elite_count: int = 5,
        tournament_size: int = 5,
        random_seed: Optional[int] = None,
    ):
        self.schedule = schedule
        self.expanded_courses = expanded_courses
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.tournament_size = tournament_size

        if random_seed is not None:
            np.random.seed(random_seed)

        self.num_courses = len(expanded_courses)
        self.num_timeslots = schedule.num_timeslots()
        self.num_classrooms = schedule.num_classrooms()

        self.temp_schedule = Schedule(
            courses=expanded_courses,
            classrooms=schedule.classrooms,
            timeslots=schedule.timeslots,
            assignments=[]
        )
        self.detector = ConflictDetector(self.temp_schedule)
        self.course_map = self.temp_schedule.get_course_map()
        self.classroom_map = schedule.get_classroom_map()

        self.valid_rooms_per_course = self._compute_valid_rooms()
        self.history = GAHistory()

    def _compute_valid_rooms(self) -> Dict[int, List[int]]:
        valid_rooms = {}
        for i, course in enumerate(self.expanded_courses):
            valid = [
                r.classroom_id for r in self.schedule.classrooms
                if r.capacity >= course.student_count
            ]
            if not valid:
                valid = [r.classroom_id for r in self.schedule.classrooms]
            valid_rooms[i] = valid
        return valid_rooms

    def _decode_chromosome(self, chromosome: np.ndarray) -> List[ScheduleAssignment]:
        assignments = []
        for i in range(self.num_courses):
            ts_id = int(chromosome[i, 0])
            room_id = int(chromosome[i, 1])
            assignments.append(ScheduleAssignment(
                course_id=i,
                timeslot_id=ts_id,
                classroom_id=room_id
            ))
        return assignments

    def _encode_chromosome(self, assignments: List[ScheduleAssignment]) -> np.ndarray:
        chrom = np.zeros((self.num_courses, 2), dtype=np.int32)
        for a in assignments:
            chrom[a.course_id, 0] = a.timeslot_id
            chrom[a.course_id, 1] = a.classroom_id
        return chrom

    def _evaluate_fitness(self, chromosome: np.ndarray) -> Tuple[float, ConflictReport]:
        assignments = self._decode_chromosome(chromosome)
        report = self.detector.detect(assignments)

        teacher_penalty = report.teacher_conflicts * 10.0
        class_penalty = report.class_conflicts * 10.0
        classroom_penalty = report.classroom_conflicts * 10.0
        capacity_penalty = report.capacity_violations * 8.0
        type_penalty = report.type_violations * 5.0

        ts_usage = np.zeros(self.num_timeslots, dtype=np.int32)
        for i in range(self.num_courses):
            ts_usage[int(chromosome[i, 0])] += 1
        variance_penalty = np.var(ts_usage) * 0.1

        total_penalty = (teacher_penalty + class_penalty + classroom_penalty +
                         capacity_penalty + type_penalty + variance_penalty)

        fitness = 1.0 / (1.0 + total_penalty)
        return fitness, report

    def _evaluate_population(self, population: np.ndarray) -> Tuple[np.ndarray, List[ConflictReport]]:
        pop_size = population.shape[0]
        fitnesses = np.zeros(pop_size, dtype=np.float64)
        reports = []
        for i in range(pop_size):
            f, r = self._evaluate_fitness(population[i])
            fitnesses[i] = f
            reports.append(r)
        return fitnesses, reports

    def _initialize_population(self, initial_assignments: List[ScheduleAssignment]) -> np.ndarray:
        population = np.zeros(
            (self.population_size, self.num_courses, 2), dtype=np.int32
        )

        if initial_assignments:
            population[0] = self._encode_chromosome(initial_assignments)
            start_idx = 1
        else:
            start_idx = 0

        for p in range(start_idx, self.population_size):
            chrom = np.zeros((self.num_courses, 2), dtype=np.int32)
            for i in range(self.num_courses):
                chrom[i, 0] = np.random.randint(0, self.num_timeslots)
                valid_rooms = self.valid_rooms_per_course[i]
                chrom[i, 1] = valid_rooms[np.random.randint(0, len(valid_rooms))]
            population[p] = chrom

        return population

    def _tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        idx = np.random.randint(0, self.population_size, self.tournament_size)
        best_idx = idx[np.argmax(fitnesses[idx])]
        return population[best_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        crossover_type = np.random.randint(0, 3)

        if crossover_type == 0:
            point = np.random.randint(1, self.num_courses)
            child1[point:] = parent2[point:]
            child2[point:] = parent1[point:]
        elif crossover_type == 1:
            p1 = np.random.randint(0, self.num_courses)
            p2 = np.random.randint(p1 + 1, self.num_courses + 1)
            child1[p1:p2] = parent2[p1:p2]
            child2[p1:p2] = parent1[p1:p2]
        else:
            for i in range(self.num_courses):
                if np.random.random() < 0.5:
                    child1[i] = parent2[i].copy()
                    child2[i] = parent1[i].copy()

        return child1, child2

    def _mutate(self, chromosome: np.ndarray, report: ConflictReport) -> np.ndarray:
        mutated = chromosome.copy()
        conflicting_indices = set()

        for c in report.conflicts:
            conflicting_indices.add(c.course_id_1)
            if c.course_id_2 >= 0:
                conflicting_indices.add(c.course_id_2)

        for i in range(self.num_courses):
            mutate_prob = self.mutation_rate
            if i in conflicting_indices:
                mutate_prob = max(mutate_prob, 0.3)

            if np.random.random() < mutate_prob:
                gene = np.random.randint(0, 2)
                if gene == 0:
                    mutated[i, 0] = np.random.randint(0, self.num_timeslots)
                else:
                    valid_rooms = self.valid_rooms_per_course[i]
                    mutated[i, 1] = valid_rooms[np.random.randint(0, len(valid_rooms))]

        return mutated

    def optimize(
        self,
        initial_assignments: List[ScheduleAssignment],
        target_conflicts: int = 0,
        patience: int = 100,
        verbose: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[List[ScheduleAssignment], ConflictReport, GAHistory]:
        start_time = time.time()

        population = self._initialize_population(initial_assignments)
        fitnesses, reports = self._evaluate_population(population)

        best_idx = np.argmax(fitnesses)
        best_chromosome = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        best_report = reports[best_idx]

        no_improvement_count = 0

        for gen in range(self.max_generations):
            sorted_idx = np.argsort(fitnesses)[::-1]
            new_population = np.zeros_like(population)

            for i in range(self.elite_count):
                new_population[i] = population[sorted_idx[i]].copy()

            new_idx = self.elite_count
            while new_idx < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)

                _, r1 = self._evaluate_fitness(child1)
                _, r2 = self._evaluate_fitness(child2)
                child1 = self._mutate(child1, r1)
                child2 = self._mutate(child2, r2)

                if new_idx < self.population_size:
                    new_population[new_idx] = child1
                    new_idx += 1
                if new_idx < self.population_size:
                    new_population[new_idx] = child2
                    new_idx += 1

            population = new_population
            fitnesses, reports = self._evaluate_population(population)

            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[current_best_idx].copy()
                best_report = reports[current_best_idx]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            avg_fit = float(np.mean(fitnesses))
            worst_fit = float(np.min(fitnesses))
            conflict_count = best_report.total_conflicts()

            self.history.generations.append(gen)
            self.history.best_fitness.append(best_fitness)
            self.history.avg_fitness.append(avg_fit)
            self.history.worst_fitness.append(worst_fit)
            self.history.conflict_counts.append(conflict_count)

            if verbose and (gen % 10 == 0 or gen == self.max_generations - 1):
                print(f"第{gen:4d}代 | 最佳适应度: {best_fitness:.6f} | "
                      f"平均适应度: {avg_fit:.6f} | 冲突数: {conflict_count}")

            if progress_callback:
                progress_callback(gen, best_fitness, avg_fit, conflict_count)

            if conflict_count <= target_conflicts:
                if verbose:
                    print(f"\n达到目标! 第{gen}代冲突数为{conflict_count}")
                break

            if no_improvement_count >= patience:
                if verbose:
                    print(f"\n连续{patience}代无改善，提前终止")
                break

        self.history.elapsed_time = time.time() - start_time

        best_assignments = self._decode_chromosome(best_chromosome)
        return best_assignments, best_report, self.history


def find_adjustments(
    old_assignments: List[ScheduleAssignment],
    new_assignments: List[ScheduleAssignment]
) -> List[Tuple[int, ScheduleAssignment, ScheduleAssignment]]:
    old_map = {a.course_id: a for a in old_assignments}
    adjustments = []
    for new_a in new_assignments:
        old_a = old_map.get(new_a.course_id)
        if old_a is None:
            adjustments.append((new_a.course_id, None, new_a))
        elif (old_a.timeslot_id != new_a.timeslot_id or
              old_a.classroom_id != new_a.classroom_id):
            adjustments.append((new_a.course_id, old_a, new_a))
    return adjustments
