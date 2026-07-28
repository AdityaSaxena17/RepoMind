import { SignUp } from "@clerk/nextjs";

export default function Page() {
  return (
    <main className="flex min-h-[calc(100svh-73px)] items-center justify-center px-6 py-12">
      <SignUp />
    </main>
  );
}
