class Logger:
    def __init__(self, fields, delimiter=",", label_interval=0):
        self.fields = fields
        self.values = {k: [] for k in self.fields}
        self._is_print = True
        self._is_file = False
        self._append = False
        self._filename = None
        self._delimiter = delimiter
        self._current_count = 0
        self._interval = label_interval

    def set_print(self, is_print):
        self._is_print = is_print

    def log_file(self, filename, append=False):
        self._filename = filename
        self._is_file = True
        self._append = append

    def register(self, fields, values):
        if isinstance(fields, (list, tuple)):
            for field, value in zip(fields, values):
                self.values[field].append(value)
        else:
            self.values[fields].append(values)

    def format(self, values):
        return self._delimiter.join([str(v) for v in values])

    def update(self):
        lens = [len(values) for values in self.values.values()]
        if max(lens) != min(lens):
            raise RuntimeError("value arrays length mismatch!")

        if self._is_print:
            self.print_stdout()

        self._current_count += 1

    def tee(self):
        self.update()

        if self._is_file:
            self.append_file()

    def last_values(self):
        return [vs[-1] for vs in self.values.values()]

    def append_file(self):
        with open(self._filename, "a") as f:
            if self._current_count == 0:
                print(self.format(self.fields), file=f)

            print(self.format(self.last_values()), file=f)

    def print_stdout(self):
        if (self._interval == 0 and self._current_count == 0) or (
            self._interval != 0 and self._current_count % self._interval == 0
        ):
            print(self.format(self.fields))

        print(self.format(self.last_values()))

    def dump_log(self, filename, header=True):
        with open(filename, "w") as f:
            print(self.format(self.fields), file=f)

            for vs in zip(*self.values.values()):
                print(self.format(vs), file=f)


if __name__ == "__main__":
    fields = ["one", "two", "three"]
    logger = Logger(fields, label_interval=4)
    logger.register("one", 1)
    logger.register("two", 1)
    logger.register("three", 1)
    logger.update()

    for i in range(10):
        logger.register(fields, [i, i + 1, i + 2])
        logger.update()

    logger.dump_log("log")
