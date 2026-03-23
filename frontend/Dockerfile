FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies based on package-lock
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the frontend source code
COPY . .

# Build Next.js application
ENV NEXT_PUBLIC_API_URL=http://localhost:8000
RUN npm run build

# Production runner stage
FROM node:18-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copy necessary build artifacts
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules

EXPOSE 3000

CMD ["npm", "start"]
