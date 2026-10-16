import { prisma } from "@repo/db";

/** Delete only the authenticated user's chat; refuse an active response. */
export async function deleteConversationForUser(
  userId: string,
  repositoryId: string,
  conversationId: string,
) {
  return prisma.$transaction(async (tx) => {
    await tx.$queryRaw`
      SELECT "id" FROM "ChatConversation"
      WHERE "id" = ${conversationId} AND "userId" = ${userId}
        AND "repositoryId" = ${repositoryId} FOR UPDATE
    `;
    const conversation = await tx.chatConversation.findFirst({
      where: { id: conversationId, userId, repositoryId },
    });
    if (!conversation) return "not_found";
    const active = await tx.chatMessage.count({
      where: { conversationId, status: { in: ["PENDING", "STREAMING"] } },
    });
    if (active > 0) return "busy";
    // Foreign-key cascades remove messages and their citations in this transaction.
    await tx.chatConversation.delete({ where: { id: conversationId } });
    return "deleted";
  });
}

/** Disconnect one account; remove unused indexed data only when no job is active. */
export async function deleteRepositoryForUser(
  userId: string,
  owner: string,
  name: string,
) {
  return prisma.$transaction(async (tx) => {
    const repository = await tx.repository.findFirst({
      where: { owner, name, users: { some: { userId } } },
    });
    if (!repository) return "not_found";
    // Match the reindex/worker lock so a job cannot start during final removal.
    await tx.$queryRaw`
      SELECT "id" FROM "Repository" WHERE "id" = ${repository.id} FOR UPDATE
    `;
    const connection = await tx.userRepository.findUnique({
      where: { userId_repositoryId: { userId, repositoryId: repository.id } },
    });
    if (!connection) return "not_found";
    const activeMessages = await tx.chatMessage.count({
      where: {
        conversation: { userId, repositoryId: repository.id },
        status: { in: ["PENDING", "STREAMING"] },
      },
    });
    if (activeMessages > 0) return "busy";
    // Count revoked connections too: they may still own chats or retained data.
    const connections = await tx.userRepository.count({
      where: { repositoryId: repository.id },
    });
    if (connections === 1) {
      const activeJobs = await tx.repositoryIndexJob.count({
        where: {
          repositoryId: repository.id,
          status: { in: ["QUEUED", "RUNNING"] },
        },
      });
      if (activeJobs > 0) return "busy";
    }
    await tx.chatConversation.deleteMany({
      where: { userId, repositoryId: repository.id },
    });
    await tx.userRepository.delete({ where: { id: connection.id } });
    if (connections === 1) {
      await tx.repository.delete({ where: { id: repository.id } });
    }
    return "deleted";
  });
}
