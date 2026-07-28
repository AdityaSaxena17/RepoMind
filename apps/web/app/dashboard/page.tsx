import { AppShell } from "../../components/app-shell";
import { RepositoryPicker } from "../../components/repository-picker";

export default async function Dashboard() {
  return (
    <AppShell activeItem="dashboard">
      <div className="px-5 py-6 md:px-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm font-medium text-sky-300">Dashboard</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-normal text-zinc-50">
              Repositories
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-400">
              Choose repositories from your GitHub App installations and click
              Connect repository to start indexing.
            </p>
          </div>
        </div>
        <div className="mt-8">
          <RepositoryPicker />
        </div>
      </div>
    </AppShell>
  );
}
