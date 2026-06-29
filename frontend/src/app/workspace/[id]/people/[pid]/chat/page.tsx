import { PersonaFullscreenChatPage } from "@/components/modules/PersonaFullscreenChatPage";

export default async function Page({
  params,
}: {
  params: Promise<{ id: string; pid: string }>;
}) {
  const { id, pid } = await params;
  return <PersonaFullscreenChatPage workspaceId={id} personId={pid} />;
}
