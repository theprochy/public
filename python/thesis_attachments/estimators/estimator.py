"""
Estimator should return criterion value calculated from estimated order.
"""
import time
from abc import abstractmethod
from typing import List

from result import Result, CompleteResult
from solution import Solution, Instance
from ulits.lazy_class import ForceClass, Namable

import sys
import threading

try:
    import thread
except ImportError:
    import _thread as thread


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


class Estimator(Namable):

    @staticmethod
    def exit_after(s):
        '''
        use as decorator to exit process if
        function takes longer than s seconds
        '''

        def outer(fn):
            def inner(*args, **kwargs):
                timer = threading.Timer(s, quit_function, args=[fn.__name__])
                timer.start()
                try:
                    result = fn(*args, **kwargs)
                finally:
                    timer.cancel()
                return result

            return inner

        return outer

    """
    Abstract class of estimator.
    """
    def time_measured_estimation(self, solution: Solution, force_estimators=None) -> Result:
        name = self.name()
        print(f'Estimator {name}')
        if not force_estimators:
            force_estimators = []
        if name not in solution.results or name in force_estimators:
            st = time.time()
            instance = self.pre_sort(solution.instance)
            try:
                result = self._estimate(instance, solution.optimal_result()).to_complete_result(self.name(), metrics=self.metrics)
                result.time = time.time() - st
            except KeyboardInterrupt:
                print("Evaluation of estimator" + self.name() + " timed out!")
                result = Result([], instance.n, 3600, [-1, -1]).to_complete_result(self.name(), metrics=self.metrics)
            solution.results[name] = result
            print(f'Evaluate estimator {name}')
            return result

        return solution.results[name]

    @abstractmethod
    def _estimate(self, instance: Instance, opt_result: CompleteResult=None) -> Result:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @abstractmethod
    def pre_sort(self, instance: Instance) -> Instance:
        """Sort instance to order in which is expected by estimator."""
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    def info(self) -> str:
        return self.name() + ': ' + self._get_info()

    @abstractmethod
    def _get_info(self) -> str:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    def __str__(self):
        return self.name()

    # @property
    # @abstractmethod
    # def normalizer(self) -> Normalizer:
    #     raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        raise NotImplementedError(f'{self.__class__} do not have implemented abstract method.')

    @classmethod
    def lazy_init(cls, *args, **kwargs):
        return ForceClass(cls, *args, **kwargs)

    def load(self):
        pass

    def name_without_normalizer(self):
        return self.name()

    def mongo_name(self):
        return self.name().replace('.', '_')
