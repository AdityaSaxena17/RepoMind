import { PrismaClient } from "@prisma/client";

declare global {
  var repoMindPrisma: PrismaClient | undefined;
}

export const prisma = globalThis.repoMindPrisma ?? new PrismaClient();

if (process.env.NODE_ENV !== "production") {
  globalThis.repoMindPrisma = prisma;
}
