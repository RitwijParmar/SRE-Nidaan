"use client";

import { useCallback, useEffect, useMemo } from "react";
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type NodeMouseHandler,
  type Node,
  type NodeTypes,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "dagre";

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
  selectedNodeId?: string | null;
  onSelectNode?: (id: string) => void;
}

function CausalNode({
  data,
}: {
  data: { label: string; role: "entry" | "junction" | "outcome"; selected: boolean };
}) {
  const theme =
    data.role === "entry"
      ? {
          border: "#85b7f5",
          borderStrong: "#2d7de4",
          badgeBorder: "#b6d5fb",
          badgeBg: "#edf5ff",
          badgeText: "#245da0",
          glow: "rgba(45, 125, 228, 0.2)",
        }
      : data.role === "outcome"
        ? {
            border: "#9acfbf",
            borderStrong: "#1f9a73",
            badgeBorder: "#c1e4d8",
            badgeBg: "#eefaf5",
            badgeText: "#1d7056",
            glow: "rgba(31, 154, 115, 0.2)",
          }
        : {
            border: "#c7d5e8",
            borderStrong: "#4f6f95",
            badgeBorder: "#dae5f3",
            badgeBg: "#f4f8fc",
            badgeText: "#466182",
            glow: "rgba(79, 111, 149, 0.18)",
          };
  const roleLabel =
    data.role === "entry"
      ? "entry"
      : data.role === "outcome"
        ? "outcome"
        : "intermediate";
  return (
    <div className="group relative">
      <Handle
        type="target"
        position={Position.Top}
        className="!h-2.5 !w-2.5 !border-[#0a7f78] !bg-[#0a7f78]"
      />
      <div
        className="min-w-[190px] cursor-pointer rounded-2xl border bg-gradient-to-br from-white to-[#eef5ff] px-5 py-3 text-center shadow-md shadow-[#122033]/10 transition-all duration-200 group-hover:shadow-lg group-hover:shadow-[#122033]/15"
        style={{
          borderColor: data.selected ? theme.borderStrong : theme.border,
          boxShadow: data.selected
            ? `0 0 0 2px ${theme.glow}, 0 12px 24px rgba(18, 32, 51, 0.14)`
            : undefined,
        }}
      >
        <span
          className="mb-1 inline-block rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
          style={{
            borderColor: theme.badgeBorder,
            background: theme.badgeBg,
            color: theme.badgeText,
          }}
        >
          {roleLabel}
        </span>
        <br />
        <span className="text-sm font-semibold text-[#122033]">{data.label}</span>
        <div className="pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-t from-[#155fd2]/5 to-transparent" />
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-2.5 !w-2.5 !border-[#0a7f78] !bg-[#0a7f78]"
      />
    </div>
  );
}

const nodeTypes: NodeTypes = {
  causal: CausalNode,
};

const NODE_WIDTH = 210;
const NODE_HEIGHT = 70;

function getLayoutedElements(
  nodes: Node[],
  edges: Edge[],
  direction: "TB" | "LR" = "TB"
): { nodes: Node[]; edges: Edge[] } {
  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({
    rankdir: direction,
    ranksep: 92,
    nodesep: 72,
    marginx: 32,
    marginy: 28,
  });

  nodes.forEach((node) => {
    graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  edges.forEach((edge) => {
    graph.setEdge(edge.source, edge.target);
  });

  dagre.layout(graph);

  const layoutedNodes = nodes.map((node) => {
    const dagreNode = graph.node(node.id);
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

function computeNodeRole(
  nodeId: string,
  edges: DAGEdgeData[]
): "entry" | "junction" | "outcome" {
  const incoming = edges.filter((edge) => edge.target === nodeId).length;
  const outgoing = edges.filter((edge) => edge.source === nodeId).length;
  if (incoming === 0 && outgoing > 0) {
    return "entry";
  }
  if (incoming > 0 && outgoing === 0) {
    return "outcome";
  }
  return "junction";
}

export default function CausalGraph({
  nodes: rawNodes,
  edges: rawEdges,
  selectedNodeId = null,
  onSelectNode,
}: CausalGraphProps) {
  const initialData = useMemo(() => {
    const rfNodes: Node[] = rawNodes.map((node) => ({
      id: node.id,
      type: "causal",
      data: {
        label: node.label,
        role: computeNodeRole(node.id, rawEdges),
        selected: selectedNodeId === node.id,
      },
      position: { x: 0, y: 0 },
      draggable: false,
      selectable: true,
    }));

    const rfEdges: Edge[] = rawEdges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      animated: edge.animated,
      style: {
        stroke: edge.animated ? "#0a7f78" : "#2b6fbe",
        strokeWidth: 2,
      },
      markerEnd: {
        type: "arrowclosed" as const,
        color: edge.animated ? "#0a7f78" : "#2b6fbe",
        width: 20,
        height: 20,
      },
    }));

    return getLayoutedElements(rfNodes, rfEdges, "TB");
  }, [rawEdges, rawNodes, selectedNodeId]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);

  useEffect(() => {
    setNodes(initialData.nodes);
    setEdges(initialData.edges);
  }, [initialData, setEdges, setNodes]);

  const onLayout = useCallback(
    (direction: "TB" | "LR") => {
      const { nodes: nextNodes, edges: nextEdges } = getLayoutedElements(nodes, edges, direction);
      setNodes(nextNodes);
      setEdges(nextEdges);
    },
    [edges, nodes, setEdges, setNodes]
  );

  const handleNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      if (onSelectNode) {
        onSelectNode(node.id);
      }
    },
    [onSelectNode]
  );

  return (
    <div className="relative h-full w-full">
      <div className="absolute right-3 top-3 z-10 flex gap-1.5">
        <button
          onClick={() => onLayout("TB")}
          className="rounded-md border border-nidaan-border bg-white px-2.5 py-1 nidaan-mono text-[10px] text-nidaan-muted transition hover:border-nidaan-accent/35 hover:text-nidaan-accent-strong"
        >
          Top-Down
        </button>
        <button
          onClick={() => onLayout("LR")}
          className="rounded-md border border-nidaan-border bg-white px-2.5 py-1 nidaan-mono text-[10px] text-nidaan-muted transition hover:border-nidaan-accent/35 hover:text-nidaan-accent-strong"
        >
          Left-Right
        </button>
      </div>
      <div className="absolute bottom-3 left-3 z-10 rounded-md border border-nidaan-border bg-white/95 px-2 py-1 text-[10px] text-nidaan-muted shadow-sm">
        Select a node to inspect linked evidence.
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.35 }}
        minZoom={0.35}
        maxZoom={1.6}
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#ceddec" gap={22} size={1} />
        <Controls position="bottom-right" showInteractive={false} />
        <MiniMap
          position="bottom-left"
          nodeColor="#0a7f78"
          maskColor="rgba(236, 243, 252, 0.86)"
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  );
}
