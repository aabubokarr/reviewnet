"use client";

import { Cloud } from "lucide-react";

const wordSets = {
  positive: [
    { text: "amazing", size: 48 }, { text: "love", size: 44 }, { text: "great", size: 40 },
    { text: "excellent", size: 36 }, { text: "smooth", size: 32 }, { text: "recommend", size: 30 },
    { text: "perfect", size: 28 }, { text: "fast", size: 26 }, { text: "beautiful", size: 24 },
    { text: "helpful", size: 22 }, { text: "easy", size: 20 }, { text: "best", size: 34 },
    { text: "fantastic", size: 28 }, { text: "intuitive", size: 22 }, { text: "reliable", size: 20 },
  ],
  neutral: [
    { text: "okay", size: 44 }, { text: "fine", size: 38 }, { text: "average", size: 34 },
    { text: "decent", size: 30 }, { text: "normal", size: 28 }, { text: "standard", size: 26 },
    { text: "basic", size: 24 }, { text: "simple", size: 22 }, { text: "usual", size: 20 },
    { text: "moderate", size: 32 }, { text: "acceptable", size: 24 }, { text: "expected", size: 20 },
  ],
  negative: [
    { text: "crash", size: 46 }, { text: "bug", size: 42 }, { text: "slow", size: 38 },
    { text: "terrible", size: 34 }, { text: "annoying", size: 32 }, { text: "waste", size: 28 },
    { text: "broken", size: 26 }, { text: "hate", size: 30 }, { text: "worst", size: 24 },
    { text: "frustrating", size: 36 }, { text: "useless", size: 22 }, { text: "laggy", size: 20 },
    { text: "disappointing", size: 28 }, { text: "glitch", size: 24 },
  ],
};

function WordCloudVis({ words, color }: { words: { text: string; size: number }[]; color: string }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2 p-6 min-h-[250px]">
      {words.map((w) => (
        <span
          key={w.text}
          className="cursor-default transition-transform hover:scale-110"
          style={{
            fontSize: `${w.size}px`,
            color,
            opacity: 0.6 + (w.size / 48) * 0.4,
            fontWeight: w.size > 30 ? 700 : 500,
          }}
        >
          {w.text}
        </span>
      ))}
    </div>
  );
}

export default function WordCloudsPage() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold">Word Clouds</h1>
        <p className="mt-1 text-muted-foreground">Visual representation of most common words in reviews.</p>
      </div>

      <div className="grid gap-6">
        <div className="glass-card rounded-xl p-5">
          <h3 className="mb-2 text-lg font-semibold text-success flex items-center gap-2">
            <Cloud className="h-5 w-5" /> Positive Sentiment
          </h3>
          <WordCloudVis words={wordSets.positive} color="hsl(160, 84%, 39%)" />
        </div>

        <div className="glass-card rounded-xl p-5">
          <h3 className="mb-2 text-lg font-semibold text-info flex items-center gap-2">
            <Cloud className="h-5 w-5" /> Neutral Sentiment
          </h3>
          <WordCloudVis words={wordSets.neutral} color="hsl(217, 91%, 60%)" />
        </div>

        <div className="glass-card rounded-xl p-5">
          <h3 className="mb-2 text-lg font-semibold text-destructive flex items-center gap-2">
            <Cloud className="h-5 w-5" /> Negative Sentiment
          </h3>
          <WordCloudVis words={wordSets.negative} color="hsl(0, 84%, 60%)" />
        </div>
      </div>
    </div>
  );
}
