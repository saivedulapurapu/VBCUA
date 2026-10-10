import { useCallback, useState } from 'react'

/** A query-string parameter kept in sync with state, so a page reload keeps it. */
export function useUrlParam(name: string): [string | null, (value: string | null) => void] {
  const [value, setValue] = useState<string | null>(() => new URLSearchParams(window.location.search).get(name))

  const update = useCallback(
    (next: string | null) => {
      const url = new URL(window.location.href)
      if (next === null) url.searchParams.delete(name)
      else url.searchParams.set(name, next)
      window.history.replaceState(null, '', url)
      setValue(next)
    },
    [name],
  )

  return [value, update]
}
