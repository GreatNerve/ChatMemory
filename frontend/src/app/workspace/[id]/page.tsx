import { WorkspaceOverviewPage } from "@/components/modules/WorkspaceOverviewPage";

export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <WorkspaceOverviewPage workspaceId={id} />;
}
