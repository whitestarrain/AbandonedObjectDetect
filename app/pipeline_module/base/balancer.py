from threading import Lock

BALANCE_CEILING_VALUE = 50

class ModuleBalancer:
    def __init__(self):
        self.max_interval = 0
        self.short_stab_interval = self.max_interval
        self.short_stab_module = None
        self.lock = Lock()
        self.ceiling_interval = 0.1

    def get_suitable_interval(self, process_interval, module):
        with self.lock:
            if module == self.short_stab_module:
                self.max_interval = (process_interval + self.max_interval) / 2
                self.short_stab_interval = module.process_interval if module.skippable else self.max_interval
                return 0
            elif process_interval > self.short_stab_interval:
                self.short_stab_module = module
                self.max_interval = process_interval
                self.short_stab_interval = module.process_interval if module.skippable else self.max_interval
                return 0
            else:
                return max(min(self.max_interval - process_interval, self.ceiling_interval), 0)
