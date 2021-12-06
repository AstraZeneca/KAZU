import logging
import statistics
import time

logger = logging.getLogger(__name__)


class Stopwatch:
    def __init__(self, call_count_interval: int = 1):
        """
        a stop watch! will produce a logging message from the Stopwatch.message().

        :param call_count_interval: when message()  is called, only log if self.call_count %
                                    self.call_count_interval == 0
        """
        self.call_count_interval = call_count_interval
        self.call_count = 0
        self.times = []

    def start(self):
        self.mark = time.time()

    def message(self, message: str):
        """
        helper function for benchmarking in logs

        :param start:
        :return:
        """

        end = time.time()
        self.call_count += 1

        time_elapsed = round(end - self.mark, 2)
        self.last_average = None
        self.times.append(time_elapsed)
        self.mark = end

        if self.call_count % self.call_count_interval == 0:
            if self.last_average is not None:
                new_average = statistics.mean(self.times)
                result = round((new_average + self.last_average) / 2, 2)
                self.last_average = new_average
                self.times = []
                logger.info(
                    f"{message}:{time_elapsed}, average: {result}, total calls:{self.call_count}, "
                )
            else:
                self.last_average = statistics.mean(self.times)
                self.times = []
