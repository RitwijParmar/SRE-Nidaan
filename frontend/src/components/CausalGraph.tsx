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
        className="!h-2.5 !w-2.5 !border-[#0b7a75] !bg-[#0b7a75]"
      />
      <div
        className={`min-w-[180px] cursor-pointer rounded-2xl border bg-gradient-to-br from-[#ffffff] to-[#eef5ff] px-5 py-3 text-center shadow-md shadow-[#122033]/10 transition-all duration-200 group-hover:border-[#0b7a75]/55 group-hover:shadow-lg group-hover:shadow-[#122033]/15 ${
          data.selected
            ? "border-[#0b7a75] ring-2 ring-[#0b7a75]/20"
            : "border-[#0e5cb5]/30"
        }`}
      >
        <span className="mb-1 inline-block rounded-full border border-[#cde1f6] bg-[#f4f9ff] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[#3f5e79]">
          {roleLabel}
        </span>
        <br />
        <span className="text-sm font-semibold text-[#122033]">{data.label}</span>
        <div className="pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-t from-[#0e5cb5]/5 to-transparent" />
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-2.5 !w-2.5 !border-[#0b7a75] !bg-[#0b7a75]"
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
        stroke: "#0b7a75",
        strokeWidth: 2,
      },
      markerEnd: {
        type: "arrowclosed" as const,
        color: "#0b7a75",
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
        <Background color="#d7e4f0" gap={22} size={1} />
        <Controls position="bottom-right" showInteractive={false} />
        <MiniMap
          position="bottom-left"
          nodeColor="#0f766e"
          maskColor="rgba(245, 240, 230, 0.75)"
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  );
}
