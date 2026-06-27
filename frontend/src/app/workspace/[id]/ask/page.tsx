import { AskPage } from "@/components/modules/AskPage";

export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <AskPage workspaceId={id} />;
}
