"use client";

import { useMemo, useCallback } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeTypes,
  Handle,
  Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "dagre";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface DAGNodeData {
  id: string;
  label: string;
}

interface DAGEdgeData {
  id: string;
  source: string;
  target: string;
  animated: boolean;
}

interface CausalGraphProps {
  nodes: DAGNodeData[];
  edges: DAGEdgeData[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom Node Component
// ─────────────────────────────────────────────────────────────────────────────

function CausalNode({ data }: { data: { label: string } }) {
  return (
    <div className="relative group">
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-blue-500 !border-blue-400 !w-2.5 !h-2.5"
      />
      <div
        className="px-5 py-3 rounded-xl bg-gradient-to-br from-[#1a2740] to-[#0f1a2e]
                    border border-blue-500/30 text-sm font-medium text-blue-100
                    shadow-lg shadow-blue-500/10 min-w-[160px] text-center
                    group-hover:border-blue-400/60 group-hover:shadow-blue-500/25
                    transition-all duration-300 cursor-default"
      >
        <span className="relative z-10">{data.label}</span>
        {/* Subtle inner glow */}
        <div className="absolute inset-0 rounded-xl bg-gradient-to-t from-blue-500/5 to-transparent" />
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-blue-500 !border-blue-400 !w-2.5 !h-2.5"
      />
    </div>
  );
}

const nodeTypes: NodeTypes = {
  causal: CausalNode,
};

// ─────────────────────────────────────────────────────────────────────────────
// Dagre Layout Engine
// ─────────────────────────────────────────────────────────────────────────────

const NODE_WIDTH = 200;
const NODE_HEIGHT = 60;

function getLayoutedElements(
  nodes: Node[],
  edges: Edge[],
  direction: "TB" | "LR" = "TB"
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));

  g.setGraph({
    rankdir: direction,
    ranksep: 80,
    nodesep: 60,
    marginx: 30,
    marginy: 30,
  });

  nodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  // Convert Dagre coordinates → React Flow coordinates
  // Dagre returns center-point coordinates; React Flow uses top-left
  const layoutedNodes = nodes.map((node) => {
    const dagreNode = g.node(node.id);
    return {
      ...node,
      position: {
        x: dagreNode.x - NODE_WIDTH / 2,
        y: dagreNode.y - NODE_HEIGHT / 2,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export default function CausalGraph({ nodes: rawNodes, edges: rawEdges }: CausalGraphProps) {
  // Transform API data → React Flow format
  const initialData = useMemo(() => {
    const rfNodes: Node[] = rawNodes.map((n) => ({
      id: n.id,
      type: "causal",
      data: { label: n.label },
      position: { x: 0, y: 0 }, // will be overridden by Dagre
    }));

    const rfEdges: Edge[] = rawEdges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      animated: e.animated,
      style: {
        stroke: "#3b82f6",
        strokeWidth: 2,
      },
      markerEnd: {
        type: "arrowclosed" as const,
        color: "#3b82f6",
        width: 20,
        height: 20,
      },
    }));

    return getLayoutedElements(rfNodes, rfEdges, "TB");
  }, [rawNodes, rawEdges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);

  const onLayout = useCallback(
    (direction: "TB" | "LR") => {
      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
        nodes,
        edges,
        direction
      );
      setNodes([...layoutedNodes]);
      setEdges([...layoutedEdges]);
    },
    [nodes, edges, setNodes, setEdges]
  );

  return (
    <div className="w-full h-full relative">
      {/* Layout toggle */}
      <div className="absolute top-3 right-3 z-10 flex gap-1.5">
        <button
          onClick={() => onLayout("TB")}
          className="px-2.5 py-1 text-[10px] font-mono rounded-md
                     bg-nidaan-card border border-nidaan-border text-nidaan-text-dim
                     hover:bg-nidaan-surface hover:text-white transition-all"
        >
          ↕ Vertical
        </button>
        <button
          onClick={() => onLayout("LR")}
          className="px-2.5 py-1 text-[10px] font-mono rounded-md
                     bg-nidaan-card border border-nidaan-border text-nidaan-text-dim
                     hover:bg-nidaan-surface hover:text-white transition-all"
        >
          ↔ Horizontal
        </button>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.3}
        maxZoom={1.5}
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
        className="bg-transparent"
      >
        <Background color="#1e293b" gap={20} size={1} />
        <Controls
          position="bottom-right"
          showInteractive={false}
        />
        <MiniMap
          position="bottom-left"
          nodeColor="#3b82f6"
          maskColor="rgba(10, 14, 26, 0.8)"
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  );
}
