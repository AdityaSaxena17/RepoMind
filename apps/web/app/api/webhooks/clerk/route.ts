import { verifyWebhook } from "@clerk/nextjs/webhooks";
import { prisma, UserStatus } from "@repo/db";
import type { NextRequest } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  try {
    const evt = await verifyWebhook(request);

    if (evt.type === "user.created" || evt.type === "user.updated") {
      const primaryEmail =
        evt.data.email_addresses.find(
          (email) => email.id === evt.data.primary_email_address_id,
        )?.email_address ??
        evt.data.email_addresses[0]?.email_address ??
        null;

      await prisma.user.upsert({
        where: {
          clerkId: evt.data.id,
        },
        create: {
          clerkId: evt.data.id,
          email: primaryEmail,
          firstName: evt.data.first_name,
          lastName: evt.data.last_name,
          username: evt.data.username,
          imageUrl: evt.data.image_url,
          status: UserStatus.ACTIVE,
          inactiveAt: null,
        },
        update: {
          email: primaryEmail,
          firstName: evt.data.first_name,
          lastName: evt.data.last_name,
          username: evt.data.username,
          imageUrl: evt.data.image_url,
          status: UserStatus.ACTIVE,
          inactiveAt: null,
        },
      });

      return Response.json({ received: true });
    }

    if (evt.type === "user.deleted") {
      if (!evt.data.id) {
        return Response.json({ received: true });
      }

      await prisma.user.updateMany({
        where: {
          clerkId: evt.data.id,
        },
        data: {
          status: UserStatus.INACTIVE,
          inactiveAt: new Date(),
        },
      });

      return Response.json({ received: true });
    }

    return Response.json({ received: true, ignored: true });
  } catch (error) {
    console.error("Clerk webhook verification failed", error);

    return new Response("Webhook verification failed", { status: 400 });
  }
}
