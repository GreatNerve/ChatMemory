import { PeoplePage } from "@/components/modules/PeoplePage";

export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <PeoplePage workspaceId={id} />;
}
