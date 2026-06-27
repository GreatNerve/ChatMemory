import { PersonDetailPage } from "@/components/modules/PersonDetailPage";

export default async function Page({
  params,
}: {
  params: Promise<{ id: string; pid: string }>;
}) {
  const { id, pid } = await params;
  return <PersonDetailPage workspaceId={id} personId={pid} />;
}
