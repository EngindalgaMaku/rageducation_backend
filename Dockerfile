# Multi-stage Dockerfile for Next.js app located in ./frontend

# 1) Dependencies layer
FROM node:20-alpine AS deps
WORKDIR /app
# Copy only package manifests from frontend
COPY frontend/package*.json ./
RUN npm ci

# 2) Build layer
FROM node:20-alpine AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
RUN npm run build

# 3) Runtime layer
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
# Cloud Run expects the app to listen on $PORT (default 8080)
ENV PORT=8080

# Copy minimal artifacts to run `next start`
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules

EXPOSE 8080
# Use the PORT env for Next.js start
CMD ["sh", "-c", "npm run start -- -p ${PORT}"]
