class Result:
    def __init__(self, order, criterion, time, bonus=None):
        self.order = order
        self.time = time
        self.criterion = criterion
        self.bonus = bonus

    def to_mongo(self):
        out = {}
        out.update(self.__dict__)
        out['empty'] = self.empty()
        return out

    def empty(self):
        return self.order is None and self.criterion is None

    def __str__(self):
        return f'(c:{self.criterion}, t:{self.time})'

    def to_complete_result(self, name, metrics=None):
        return CompleteResult(self.order, self.criterion, self.time, estimator_name=name, metrics=metrics, bonus=self.bonus)


class CompleteResult(Result):
    def __init__(self, order, criterion, time, comment='', metrics=None, bonus=None, estimator_name=None, **kwargs):
        super().__init__(order, criterion, time)
        self.estimator_name = estimator_name
        self.comment = comment
        self.bonus = bonus
        self.metrics = metrics
        self.one = 1

    def __str__(self):
        return f'(c:{self.criterion}, t:{self.time})'

    def __repr__(self):
        return str(self)

    def to_complete_result(self, name, metrics=None):
        return CompleteResult(self.order, self.criterion, self.time, comment=self.comment, bonus=self.bonus,
                              estimator_name=name, metrics=metrics)


ZERO_RESULT = Result(0, 0, 0)
