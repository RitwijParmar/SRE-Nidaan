# ── Dockerfile for The Face (Next.js Frontend) ──
FROM node:18-alpine AS builder

WORKDIR /app

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
