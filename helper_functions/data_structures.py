"""
========================================================================================================================
===                                                                                                                  ===
=== Definitions of special datastructures to be used                                                                 ===
===                                                                                                                  ===
========================================================================================================================
"""


class Heap(list):
    """
    ====================================================================================================================
    Zero-indexed list-based min-or-max heap class
    ====================================================================================================================
    Code skeleton by Bart Gerritsen, EEMCS, TU Delft; from course TI3111TU - Algorithms and Data Structures 2019/2020
    Implementation by Josephine Pockelé
    """
    # distinguish between ordering types ...
    MIN_HEAP, MAX_HEAP = 0, 1

    @staticmethod
    def print_heap(a_heap, title=''):
        """
        print heap as a string
        """
        print('{:s} {:s}'.format(title, str(a_heap)))

    def __init__(self, hp_type=MIN_HEAP, items=None):
        """
        init a heap as a list
        """
        super().__init__(self)
        # register the heap ordering ...
        self.type = hp_type
        # if items given, preload on heap ...
        if items:
            self.extend(items)
        # install heap on list ...
        self.heapify()

    def __str__(self):
        """
        return list as string
        """
        _items = ', '.join([str(_item) for _item in self])
        return _items

    def empty(self):
        """
        return T|F heap empty
        """
        return not bool(self.size())

    def size(self):
        """
        return current heap size
        """
        return len(self)

    def push_heap(self, item):
        """
        push new item on the heap
        """
        # append the item ...
        self.append(item)
        # bubble up ...
        self._bubble_up(self.size() - 1)

    def pop_heap(self):
        """
        pop from the top of the heap
        """
        # pop heap top [0] and swap with last ...
        item, self[0] = self[0], self[-1]
        # remove the double entry ...
        self.pop(-1)
        # bubble down ...
        self._bubble_down(0)

        return item

    def in_order(self, child, parent):
        """
        return T|F parent and child are in heap-order
        """
        if parent is None or child is None:
            return True

        if not self.type:
            return self[parent] <= self[child]

        elif self.type:
            return self[parent] >= self[child]

    def _bubble_up(self, child):
        """
        swap last inserted element into correct position
        """
        # swap child-parent until parent in order ...
        parent = (child - 1) // 2
        # What is going on? - test line
        # print(child, self[child], parent, self[parent], not self.in_order(child, parent) and parent != -1)
        if not self.in_order(child, parent) and parent != -1:
            self[parent], self[child] = self[child], self[parent]
            self._bubble_up(parent)

    def _bubble_down(self, parent):
        """
        swap parent downwards into correct position
        """

        def child_to_swap(lchld, rchld, sze):
            """
            return index of child to involve in swapping, None if no child
            """
            chld = None

            # Right child exists?
            if rchld in range(sze):
                # Would a swap of parent and rchild maintain the heap?
                if self.in_order(lchld, rchld):
                    chld = rchld
                # Would a swap of parent and lchild maintain the heap?
                elif self.in_order(rchld, lchld):
                    chld = lchld

            elif lchld in range(sze - 1):
                chld = lchld

            return chld

        if self.empty():
            return

        size = self.size()
        lchild = 2 * parent + 1
        rchild = lchild + 1
        swp_ndx = child_to_swap(lchild, rchild, size)

        if swp_ndx is not None:
            if not self.in_order(swp_ndx, parent):
                swap = (self[swp_ndx], self[parent])
                self[parent], self[swp_ndx] = swap
                self._bubble_down(swp_ndx)

    def heapify(self):
        """
        install heap property on list
        """
        size = self.size()
        # heap tree leaves have no children and
        # already have heap property ...
        # skip leaves ...
        for ndx in range(size // 2, -1, -1):
            self._bubble_down(ndx)
