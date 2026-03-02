"use client";

interface Word {
  text: string;
  size: number;
}

interface WordCloudProps {
  words: Word[];
  colors?: string[];
  color?: string;
}

export function WordCloud({ words, colors, color }: WordCloudProps) {
  const displayColors = colors || (color ? [color] : ["currentColor"]);
  
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 p-4 min-h-[150px]">
      {words.map((w, i) => (
        <span
          key={w.text}
          className="cursor-default transition-transform hover:scale-110"
          style={{
            fontSize: `${w.size / 1.5}px`, // Scaled down for dashboard
            color: displayColors[i % displayColors.length],
            fontWeight: w.size > 30 ? 700 : 500,
          }}
        >
          {w.text}
        </span>
      ))}
    </div>
  );
}
