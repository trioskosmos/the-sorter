import { useMemo } from 'react';
import { useSongData } from './useSongData';
import type { Song } from '~/types';

export const useSongMap = () => {
  const songs = useSongData();

  const songMap = useMemo(() => {
    const map = new Map<string, Song>();
    if (Array.isArray(songs)) {
      songs.forEach((song) => {
        // song.id might be number or string, ensure it is string for the key
        map.set(String(song.id), song);
      });
    }
    return map;
  }, [songs]);

  return songMap;
};
