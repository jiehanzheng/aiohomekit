"""A manually advanced clock for subscription timers, leaving network timeouts real."""


class ControlledTimer:
    def __init__(self, when, callback, args):
        self.deadline = when
        self.callback = callback
        self.args = args
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class ControlledLoop:
    def __init__(self):
        self.now = 100.0
        self.timers = []

    def time(self):
        return self.now

    def call_at(self, when, callback, *args):
        timer = ControlledTimer(when, callback, args)
        self.timers.append(timer)
        return timer

    def advance(self):
        pending = [timer for timer in self.timers if not timer.cancelled]
        assert len(pending) == 1
        timer = pending[0]
        self.now = timer.deadline
        timer.cancelled = True
        timer.callback(*timer.args)
