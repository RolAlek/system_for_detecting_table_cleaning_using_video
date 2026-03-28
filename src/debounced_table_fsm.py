from src.domain import TableEvent, TableEventKind


class DebouncedTableStateMachine:
    """
    Logical state of the table with debounce by "raw" frames.
    Accumulates events: empty, occupied, approach (after empty).
    """

    def __init__(self, debounce_frames: int) -> None:
        self._debounce_frames = debounce_frames

        self._logical_occupied: bool | None = None
        self._raw_streak: int = 0
        self._last_raw: bool | None = None
        self._events: list[TableEvent] = []

    @property
    def events(self) -> list[TableEvent]:
        """Get the list of events."""
        return self._events

    @property
    def logical_occupied(self) -> bool | None:
        """Get the current logical state of the table."""
        return self._logical_occupied

    def update(self, frame_idx: int, fps: float, raw_person_in_zone: bool) -> None:
        """Update the FSM with a new frame."""
        self._advance_raw_streak(raw_person_in_zone)

        if self._logical_occupied is None:
            self._try_commit_initial_logical_state(frame_idx, fps)
            return

        self._try_commit_transition(frame_idx, fps, raw_person_in_zone)

    def _advance_raw_streak(self, raw: bool) -> None:
        """Counts how many frames in a row the detector gives the same raw value."""
        if self._last_raw is None:
            self._last_raw = raw
            self._raw_streak = 1

        elif raw == self._last_raw:
            self._raw_streak += 1

        else:
            self._last_raw = raw
            self._raw_streak = 1

    def _append_event(self, frame_idx: int, fps: float, kind: TableEventKind) -> None:
        """Append a new event to the list."""
        self._events.append(TableEvent.at_frame(frame_idx, fps, kind))

    def _try_commit_initial_logical_state(self, frame_idx: int, fps: float) -> None:
        """The first stable raw value sets the initial "table occupied / empty"."""

        if self._raw_streak < self._debounce_frames or self._last_raw is None:
            return

        self._logical_occupied = self._last_raw

        if self._logical_occupied:
            self._append_event(frame_idx, fps, TableEventKind.OCCUPIED)
        else:
            self._append_event(frame_idx, fps, TableEventKind.EMPTY)

    def _try_commit_transition(
        self, frame_idx: int, fps: float, raw_person_in_zone: bool
    ) -> None:
        """Change of logical state after stable difference of raw from the current logic."""

        if (
            self._raw_streak < self._debounce_frames
            or raw_person_in_zone == self._logical_occupied
        ):
            return

        self._logical_occupied = raw_person_in_zone

        if self._logical_occupied:
            self._append_event(frame_idx, fps, TableEventKind.OCCUPIED)
            self._append_event(frame_idx, fps, TableEventKind.APPROACH)
        else:
            self._append_event(frame_idx, fps, TableEventKind.EMPTY)
