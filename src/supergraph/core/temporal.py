
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


class DateRange(NamedTuple):
    start_ms: int
    end_ms: int

    @property
    def mid_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


_PAT_DMY = re.compile(
    r'\b(0?[1-9]|[12]\d|3[01])\s+(january|february|march|april|may|june|july|august|'
    r'september|october|november|december),?\s+(\d{4})\b', re.I
)

_PAT_MDY = re.compile(
    r'\b(january|february|march|april|may|june|july|august|'
    r'september|october|november|december)\s+(?:(\d{1,2}),?\s+)?(\d{4})\b', re.I
)

_PAT_ISO = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2})(?::(\d{2}))?)?\b')

_PAT_ISO_MONTH = re.compile(r'\b(\d{4})-(\d{2})\b')

_PAT_YEAR = re.compile(r'\b((?:19|20)\d{2})\b')

_PAT_LOCOMO = re.compile(
    r'(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+'
    r'(january|february|march|april|may|june|july|august|'
    r'september|october|november|december),?\s+(\d{4})', re.I
)


_PAT_AGO = re.compile(
    r'\b(?:(\d+)|a|an)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b', re.I
)

_PAT_LAST = re.compile(
    r'\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
    r'week|month|year|january|february|march|april|may|june|july|august|'
    r'september|october|november|december)\b', re.I
)

_PAT_BEFORE_AFTER = re.compile(
    r'\b(before|after|since|until)\s+(january|february|march|april|may|june|july|august|'
    r'september|october|november|december)\s+(\d{4})\b', re.I
)

_PAT_RELATIVE_TO = re.compile(
    r'\bthe\s+(week|day|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+'
    r'before\s+(\d{1,2})\s+(january|february|march|april|may|june|july|august|'
    r'september|october|november|december),?\s+(\d{4})\b', re.I
)


def parse_date(text: str, reference: datetime | None = None) -> int | None:
    if reference is None:
        reference = datetime.now(timezone.utc)

    text = text.strip()

    if text.isdigit() and len(text) > 8:
        return int(text)

    m = _PAT_LOCOMO.search(text)
    if m:
        try:
            hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3).lower()
            day = int(m.group(4))
            month = _MONTHS[m.group(5).lower()]
            year = int(m.group(6))
            if ampm == "pm" and hour != 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            return _to_ms(dt)
        except (ValueError, OverflowError):
            pass

    m = _PAT_ISO.search(text)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            h = int(m.group(4) or 0)
            mi = int(m.group(5) or 0)
            s = int(m.group(6) or 0)
            dt = datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
            return _to_ms(dt)
        except (ValueError, OverflowError):
            pass

    m = _PAT_DMY.search(text)
    if m:
        try:
            d, mo_name, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            dt = datetime(y, _MONTHS[mo_name], d, tzinfo=timezone.utc)
            return _to_ms(dt)
        except (ValueError, OverflowError):
            pass

    m = _PAT_MDY.search(text)
    if m:
        try:
            mo_name = m.group(1).lower()
            d = int(m.group(2)) if m.group(2) else 1
            y = int(m.group(3))
            dt = datetime(y, _MONTHS[mo_name], d, tzinfo=timezone.utc)
            return _to_ms(dt)
        except (ValueError, OverflowError):
            pass

    m = _PAT_ISO_MONTH.search(text)
    if m:
        try:
            y, mo = int(m.group(1)), int(m.group(2))
            dt = datetime(y, mo, 1, tzinfo=timezone.utc)
            return _to_ms(dt)
        except (ValueError, OverflowError):
            pass

    m = _PAT_YEAR.search(text)
    if m:
        prefix = text[:m.start()].strip().split()
        if prefix and len(prefix[-1]) > 2 and prefix[-1].lower() not in _MONTHS:
            pass
        else:
            y = int(m.group(1))
            dt = datetime(y, 1, 1, tzinfo=timezone.utc)
            return _to_ms(dt)

    m = _PAT_AGO.search(text)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        unit = m.group(2).lower()
        delta = _unit_to_timedelta(n, unit)
        dt = reference - delta
        return _to_ms(dt)

    m = _PAT_LAST.search(text)
    if m:
        token = m.group(1).lower()
        dt = _resolve_last(token, reference)
        if dt:
            return _to_ms(dt)

    return None


def parse_date_range(text: str, reference: datetime | None = None) -> DateRange | None:
    if reference is None:
        reference = datetime.now(timezone.utc)

    text = text.strip()

    m = _PAT_RELATIVE_TO.search(text)
    if m:
        unit = m.group(1).lower()
        d, mo_name, y = int(m.group(2)), m.group(3).lower(), int(m.group(4))
        anchor = datetime(y, _MONTHS[mo_name], d, tzinfo=timezone.utc)
        if unit == "week":
            start = anchor - timedelta(days=7)
            return DateRange(_to_ms(start), _to_ms(anchor))
        elif unit == "day":
            prev = anchor - timedelta(days=1)
            return DateRange(_to_ms(_start_of_day(prev)), _to_ms(_end_of_day(prev)))
        elif unit in _WEEKDAYS:
            target_wd = _WEEKDAYS[unit]
            days_back = (anchor.weekday() - target_wd) % 7
            if days_back == 0:
                days_back = 7
            target = anchor - timedelta(days=days_back)
            return DateRange(_to_ms(_start_of_day(target)), _to_ms(_end_of_day(target)))

    m = _PAT_BEFORE_AFTER.search(text)
    if m:
        direction = m.group(1).lower()
        mo_name = m.group(2).lower()
        y = int(m.group(3))
        anchor = datetime(y, _MONTHS[mo_name], 1, tzinfo=timezone.utc)
        if direction in ("before", "until"):
            return DateRange(0, _to_ms(anchor))
        else:
            return DateRange(_to_ms(anchor), _to_ms(reference + timedelta(days=3650)))

    ms = parse_date(text, reference)
    if ms is not None:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        if _PAT_YEAR.fullmatch(text.strip()):
            start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
            end = datetime(dt.year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return DateRange(_to_ms(start), _to_ms(end))
        m = _PAT_ISO_MONTH.fullmatch(text.strip())
        if m or (not _PAT_DMY.search(text) and _PAT_MDY.search(text) and not _PAT_MDY.search(text).group(2)):
            import calendar
            _, last_day = calendar.monthrange(dt.year, dt.month)
            end = datetime(dt.year, dt.month, last_day, 23, 59, 59, tzinfo=timezone.utc)
            return DateRange(ms, _to_ms(end))
        return DateRange(
            _to_ms(_start_of_day(dt)),
            _to_ms(_end_of_day(dt)),
        )

    return None


def extract_dates(text: str, reference: datetime | None = None) -> list[int]:
    if reference is None:
        reference = datetime.now(timezone.utc)

    seen_spans: set[tuple[int, int]] = set()
    results: list[int] = []

    def _overlaps(start: int, end: int) -> bool:
        for s, e in seen_spans:
            if start < e and end > s:
                return True
        return False

    def _add(ms: int, match_start: int, match_end: int) -> None:
        if not _overlaps(match_start, match_end):
            seen_spans.add((match_start, match_end))
            results.append(ms)

    for pattern, handler in [
        (_PAT_LOCOMO, lambda m: _to_ms(datetime(
            int(m.group(6)), _MONTHS[m.group(5).lower()], int(m.group(4)),
            int(m.group(1)) + (12 if m.group(3).lower() == 'pm' and int(m.group(1)) != 12 else 0),
            int(m.group(2)), tzinfo=timezone.utc))),
        (_PAT_DMY, lambda m: _to_ms(datetime(
            int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)), tzinfo=timezone.utc))),
        (_PAT_MDY, lambda m: _to_ms(datetime(
            int(m.group(3)), _MONTHS[m.group(1).lower()],
            int(m.group(2)) if m.group(2) else 1, tzinfo=timezone.utc))),
        (_PAT_ISO, lambda m: _to_ms(datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc))),
    ]:
        for m in pattern.finditer(text):
            try:
                _add(handler(m), m.start(), m.end())
            except (ValueError, TypeError):
                continue

    for m in _PAT_AGO.finditer(text):
        try:
            n = int(m.group(1)) if m.group(1) else 1
            unit = m.group(2).lower()
            delta = _unit_to_timedelta(n, unit)
            _add(_to_ms(reference - delta), m.start(), m.end())
        except (ValueError, TypeError):
            continue

    for m in _PAT_LAST.finditer(text):
        try:
            token = m.group(1).lower()
            dt = _resolve_last(token, reference)
            if dt:
                _add(_to_ms(dt), m.start(), m.end())
        except (ValueError, TypeError):
            continue

    return results


def _unit_to_timedelta(n: int, unit: str) -> timedelta:
    unit = unit.lower()
    if unit == "second":
        return timedelta(seconds=n)
    if unit == "minute":
        return timedelta(minutes=n)
    if unit == "hour":
        return timedelta(hours=n)
    if unit == "day":
        return timedelta(days=n)
    if unit == "week":
        return timedelta(weeks=n)
    if unit == "month":
        return timedelta(days=n * 30)
    if unit == "year":
        return timedelta(days=n * 365)
    return timedelta(days=n)


def _resolve_last(token: str, reference: datetime) -> datetime | None:
    if token in _WEEKDAYS:
        target_wd = _WEEKDAYS[token]
        days_back = (reference.weekday() - target_wd) % 7
        if days_back == 0:
            days_back = 7
        return reference - timedelta(days=days_back)
    if token == "week":
        return reference - timedelta(weeks=1)
    if token == "month":
        m = reference.month - 1 or 12
        y = reference.year if reference.month > 1 else reference.year - 1
        return reference.replace(year=y, month=m, day=1)
    if token == "year":
        return reference.replace(year=reference.year - 1, month=1, day=1)
    if token in _MONTHS:
        mo = _MONTHS[token]
        y = reference.year if mo < reference.month else reference.year - 1
        return datetime(y, mo, 1, tzinfo=timezone.utc)
    return None
