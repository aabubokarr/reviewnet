"use client";

import { useState } from "react";
import { 
  FileText, 
  Plus, 
  Tag as TagIcon, 
  FolderPlus, 
  ChevronRight,
  Search,
  Highlighter,
  Trash2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Document {
  id: string;
  name: string;
  content: string;
}

interface Code {
  id: string;
  name: string;
  color: string;
  selections: Selection[];
}

interface Selection {
  id: string;
  docId: string;
  text: string;
  startIndex: number;
  endIndex: number;
}

interface Theme {
  id: string;
  name: string;
  codes: string[]; // code IDs
}

export default function ManualAnalysisPage() {
  const [documents, setDocuments] = useState<Document[]>([
    { id: "1", name: "User_Feedback_A.txt", content: "The user interface is very intuitive. However, the performance dips when loading large datasets. I love the new dark mode, it looks sleek." },
    { id: "2", name: "Interview_Notes_01.txt", content: "Participant mentioned that they found the dashboard useful for quick overview but struggled with deep dive analysis. They suggested adding more filters." }
  ]);
  const [activeDocId, setActiveDocId] = useState<string>("1");
  const [codes, setCodes] = useState<Code[]>([
    { id: "c1", name: "UI/UX", color: "bg-blue-500/20 text-blue-700 border-blue-200", selections: [] },
    { id: "c2", name: "Performance", color: "bg-orange-500/20 text-orange-700 border-orange-200", selections: [] },
  ]);
  const [selections, setSelections] = useState<Selection[]>([]);
  const [themes, setThemes] = useState<Theme[]>([
    { id: "t1", name: "User Experience", codes: ["c1"] },
  ]);
  
  const [currentSelection, setCurrentSelection] = useState<string>("");

  const activeDoc = documents.find(d => d.id === activeDocId);

  const handleTextSelection = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim().length > 0) {
      setCurrentSelection(selection.toString());
    } else {
      setCurrentSelection("");
    }
  };

  const addSelection = (codeId: string) => {
    if (!currentSelection) return;
    
    const newSelection: Selection = {
      id: Math.random().toString(36).substr(2, 9),
      docId: activeDocId,
      text: currentSelection,
      startIndex: 0, // Placeholder
      endIndex: 0,   // Placeholder
    };

    setSelections([...selections, newSelection]);
    
    setCodes(codes.map(c => 
      c.id === codeId ? { ...c, selections: [...c.selections, newSelection] } : c
    ));
    
    setCurrentSelection("");
  };

  return (
    <div className="flex flex-col h-[calc(100vh-100px)] animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">Manual Thematic Analysis</h1>
          <p className="mt-1 text-muted-foreground">Extract themes and patterns from your qualitative data.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="gap-2">
            <Plus className="h-4 w-4" /> Upload Doc
          </Button>
          <Button size="sm" className="gradient-primary">Export Results</Button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6 flex-1 min-h-0">
        {/* Left Panel: Documents */}
        <div className="col-span-3 glass-card rounded-xl flex flex-col min-h-0">
          <div className="p-4 border-b border-border flex items-center justify-between bg-accent/10">
            <h3 className="font-semibold flex items-center gap-2 text-sm uppercase tracking-wider">
              <FileText className="h-4 w-4 text-primary" /> Documents
            </h3>
            <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full font-bold">
              {documents.length}
            </span>
          </div>
          <div className="p-2 flex-1 overflow-y-auto space-y-1">
            {documents.map((doc) => (
              <button
                key={doc.id}
                onClick={() => setActiveDocId(doc.id)}
                className={cn(
                  "w-full text-left p-3 rounded-lg text-sm transition-all flex items-center justify-between group",
                  activeDocId === doc.id 
                    ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20" 
                    : "hover:bg-accent/50 text-muted-foreground"
                )}
              >
                <div className="flex items-center gap-3 truncate">
                  <FileText className={cn("h-4 w-4 shrink-0", activeDocId === doc.id ? "text-primary-foreground" : "text-muted-foreground")} />
                  <span className="truncate font-medium">{doc.name}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Middle Panel: Workspace */}
        <div className="col-span-6 glass-card rounded-xl flex flex-col min-h-0 relative bg-background/50">
          <div className="p-4 border-b border-border flex items-center justify-between bg-accent/5">
            <div className="flex items-center gap-2">
              <Highlighter className="h-4 w-4 text-primary" />
              <h3 className="font-semibold truncate max-w-[200px] text-sm uppercase tracking-wider">{activeDoc?.name}</h3>
            </div>
          </div>
          <div 
            className="p-10 flex-1 overflow-y-auto leading-relaxed text-lg font-serif selection:bg-primary/20"
            onMouseUp={handleTextSelection}
          >
            {activeDoc?.content.split('\n').map((line, i) => (
              <p key={i} className="mb-4">{line}</p>
            ))}
          </div>
          
          {currentSelection && (
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-popover/95 backdrop-blur-md border border-border shadow-2xl rounded-2xl px-6 py-4 flex flex-col gap-3 animate-in fade-in zoom-in-95 slide-in-from-bottom-4 min-w-[300px]">
              <div className="flex flex-col gap-1 border-b border-border pb-2">
                <p className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest">Selected Text</p>
                <p className="text-sm italic line-clamp-2">&quot;{currentSelection}&quot;</p>
              </div>
              <div className="flex flex-col gap-2">
                <p className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest">Apply Code</p>
                <div className="flex flex-wrap gap-2">
                  {codes.map(code => (
                    <button 
                      key={code.id}
                      onClick={() => addSelection(code.id)}
                      className={cn("px-3 py-1.5 rounded-full text-[11px] font-bold border transition-all hover:scale-105 shadow-sm active:scale-95", code.color)}
                    >
                      {code.name}
                    </button>
                  ))}
                  <button className="h-7 w-7 rounded-full bg-accent flex items-center justify-center hover:bg-accent-foreground/20 transition-colors">
                    <Plus className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel: Codes & Themes */}
        <div className="col-span-3 space-y-4 flex flex-col min-h-0">
          <div className="glass-card rounded-xl flex flex-col min-h-0 flex-1 bg-background/50">
            <div className="p-4 border-b border-border flex items-center justify-between bg-accent/10">
              <h3 className="font-semibold flex items-center gap-2 text-sm uppercase tracking-wider">
                <TagIcon className="h-4 w-4 text-primary" /> Codes
              </h3>
              <Button variant="ghost" size="icon" className="h-7 w-7 rounded-full hover:bg-primary/10 hover:text-primary"><Plus className="h-4 w-4" /></Button>
            </div>
            <div className="p-3 overflow-y-auto space-y-3">
              {codes.map(code => (
                <div key={code.id} className={cn("p-4 rounded-xl border flex flex-col gap-2 shadow-sm transition-all hover:shadow-md", code.color)}>
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm tracking-tight">{code.name}</span>
                    <span className="text-[10px] font-black uppercase text-primary/60">{code.selections.length} Inst</span>
                  </div>
                  {code.selections.length > 0 && (
                    <div className="mt-1 flex flex-col gap-1">
                      {code.selections.slice(-2).map(s => (
                        <p key={s.id} className="text-[10px] italic line-clamp-1 border-l border-primary/20 pl-2 text-muted-foreground">
                          {s.text}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="glass-card rounded-xl flex flex-col h-2/5 min-h-0 bg-background/50">
            <div className="p-4 border-b border-border flex items-center justify-between bg-accent/10">
              <h3 className="font-semibold flex items-center gap-2 text-sm uppercase tracking-wider">
                <FolderPlus className="h-4 w-4 text-primary" /> Themes
              </h3>
              <Button variant="ghost" size="icon" className="h-7 w-7 rounded-full hover:bg-primary/10 hover:text-primary"><Plus className="h-4 w-4" /></Button>
            </div>
            <div className="p-4 overflow-y-auto space-y-3">
              {themes.length === 0 && (
                <div className="text-center py-10">
                  <p className="text-sm text-muted-foreground italic">No themes categorized.</p>
                </div>
              )}
              {themes.map(theme => (
                <div key={theme.id} className="p-4 rounded-xl bg-accent/5 border border-border/50 group hover:border-primary/30 transition-all">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-bold text-sm tracking-tight">{theme.name}</span>
                    <Trash2 className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-destructive cursor-pointer transition-colors" />
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {theme.codes.map(cid => {
                      const c = codes.find(x => x.id === cid);
                      return c ? (
                        <span key={cid} className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-background border border-border shadow-sm">
                          {c.name}
                        </span>
                      ) : null;
                    })}
                    <button className="px-2 py-0.5 rounded-md text-[10px] font-bold border border-dashed border-border hover:bg-accent flex items-center gap-1 group/add">
                      <Plus className="h-2 w-2" /> Assign
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
