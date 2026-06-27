import {
  Document,
  Page,
  StyleSheet,
  Text,
  View,
  pdf,
} from "@react-pdf/renderer";
import { ChatMessage } from "@/lib/api/types";

const styles = StyleSheet.create({
  page: {
    backgroundColor: "#0a0a0a",
    color: "#f5f5f5",
    padding: 32,
    fontFamily: "Helvetica",
  },
  header: {
    borderBottomWidth: 2,
    borderBottomColor: "#f5f5f5",
    paddingBottom: 12,
    marginBottom: 20,
  },
  title: {
    fontSize: 14,
    fontFamily: "Courier",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 9,
    fontFamily: "Courier",
    color: "#a3a3a3",
    marginTop: 4,
    textTransform: "uppercase",
  },
  messages: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  userBubble: {
    alignSelf: "flex-end",
    maxWidth: "85%",
    backgroundColor: "#1a1a1a",
    borderWidth: 2,
    borderColor: "#f5f5f5",
    padding: 10,
  },
  assistantBubble: {
    alignSelf: "flex-start",
    maxWidth: "85%",
    borderLeftWidth: 4,
    borderLeftColor: "#e8ff00",
    paddingLeft: 10,
    paddingVertical: 4,
  },
  label: {
    fontSize: 8,
    fontFamily: "Courier",
    color: "#a3a3a3",
    textTransform: "uppercase",
    marginBottom: 4,
  },
  body: {
    fontSize: 10,
    lineHeight: 1.45,
    color: "#f5f5f5",
  },
  footer: {
    position: "absolute",
    bottom: 24,
    left: 32,
    right: 32,
    fontSize: 8,
    fontFamily: "Courier",
    color: "#444444",
    textTransform: "uppercase",
  },
});

function ChatPdfDocument({
  workspaceName,
  displayName,
  history,
}: {
  workspaceName: string;
  displayName: string;
  history: ChatMessage[];
}) {
  const exportedAt = new Date().toLocaleString();

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        <View style={styles.header}>
          <Text style={styles.title}>{displayName} — Persona chat</Text>
          <Text style={styles.subtitle}>{workspaceName} · exported {exportedAt}</Text>
        </View>

        <View style={styles.messages}>
          {history
            .filter((m) => m.content.trim())
            .map((m, i) =>
              m.role === "user" ? (
                <View key={`u-${i}`} style={styles.userBubble}>
                  <Text style={styles.label}>You</Text>
                  <Text style={styles.body}>{m.content}</Text>
                </View>
              ) : (
                <View key={`a-${i}`} style={styles.assistantBubble}>
                  <Text style={styles.label}>{displayName}</Text>
                  <Text style={styles.body}>{m.content}</Text>
                </View>
              ),
            )}
        </View>

        <Text style={styles.footer} fixed>
          ChatMemory persona export
        </Text>
      </Page>
    </Document>
  );
}

/** Generate and trigger download of persona chat history as a styled PDF. */
export async function downloadPersonaChatPdf(opts: {
  workspaceName: string;
  displayName: string;
  history: ChatMessage[];
}): Promise<void> {
  const blob = await pdf(
    <ChatPdfDocument
      workspaceName={opts.workspaceName}
      displayName={opts.displayName}
      history={opts.history}
    />,
  ).toBlob();

  const safeName = opts.displayName.replace(/[^\w.-]+/g, "_").slice(0, 40);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `persona-chat-${safeName}.pdf`;
  anchor.click();
  URL.revokeObjectURL(url);
}
