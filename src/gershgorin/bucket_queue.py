class BucketQueue:
    """A BucketQueue is a priority queue consisting of C many buckets for priorities from 0 to C."""

    def __init__(self, sets):
        self.sets = sets.copy()
        self.len = len(sets)
        # init queue
        self.queue = {prio: set() for prio in range(self.len)}
        self.build_queue(sets)
        self.top = None

    def __len__(self):
        return self.len

    def build_queue(self, sets):
        """Populates the queue."""
        for i in range(self.len):
            # priority is equal to number of uncovered nodes per set
            num_uncovered = len(sets[i])
            self.insert(num_uncovered, i)

    def insert(self, prio, value):
        """Inserts a value with a priority into the queue."""
        self.queue[prio].add(value)

    def extract_max(self):
        """Extracts an arbitrary element with maximum priority."""
        max_idx = self.len-1 if self.top is None else self.top
        for prio in range(max_idx, 0, -1):
            if len(self.queue[prio]) > 0:
                self.top = prio
                return prio, self.queue[prio].pop()

    def delete(self, i):
        """Finds element i in the queue and returns its priority."""
        for prio in range(self.len):
            if i in self.queue[prio]:
                self.queue[prio].remove(i)
                return prio, i
