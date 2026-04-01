# ── Dockerfile for The Face (Next.js Frontend) ──
FROM node:18-alpine AS builder

WORKDIR /app

ARG NEXT_PUBLIC_API_URL=http://localhost:8001
ARG NEXT_PUBLIC_CANONICAL_HOST=
ARG NEXT_PUBLIC_BIND_ALIAS_HOST=
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_CANONICAL_HOST=${NEXT_PUBLIC_CANONICAL_HOST}
ENV NEXT_PUBLIC_BIND_ALIAS_HOST=${NEXT_PUBLIC_BIND_ALIAS_HOST}

# Install dependencies
COPY frontend/package*.json ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# Copy application code
COPY frontend/ .

# Build Next.js application
RUN npm run build

# ── Production Image ──
FROM node:18-alpine AS runner

WORKDIR /app

ARG NEXT_PUBLIC_API_URL=http://localhost:8001
ARG NEXT_PUBLIC_CANONICAL_HOST=
ARG NEXT_PUBLIC_BIND_ALIAS_HOST=
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_CANONICAL_HOST=${NEXT_PUBLIC_CANONICAL_HOST}
ENV NEXT_PUBLIC_BIND_ALIAS_HOST=${NEXT_PUBLIC_BIND_ALIAS_HOST}

# Copy built assets
COPY --from=builder /app/package.json ./
COPY --from=builder /app/package-lock.json ./package-lock.json
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules

# Expose Next.js port
EXPOSE 3000

# Start server
CMD ["npm", "start"]
