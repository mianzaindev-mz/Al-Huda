document.addEventListener("DOMContentLoaded", () => {
    const textarea = document.querySelector("textarea");
    const sendButton = document.querySelector("button.w-10.h-10");
    const mainSection = document.querySelector("section");

    if (!textarea || !sendButton) {
        console.error("UI elements not found");
        return;
    }

    // Create chat container dynamically
    const chatContainer = document.createElement("div");
    chatContainer.className =
        "w-full max-w-4xl mx-auto flex flex-col space-y-4 mt-8";
    mainSection.innerHTML = "";
    mainSection.appendChild(chatContainer);

    function addMessage(text, sender) {
        const msg = document.createElement("div");

        const base =
            "px-5 py-3 rounded-2xl max-w-[80%] whitespace-pre-wrap leading-relaxed";

        if (sender === "user") {
            msg.className =
                base +
                " self-end bg-primary text-white shadow-md";
        } else {
            msg.className =
                base +
                " self-start bg-slate-100 dark:bg-zinc-900 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-800";
        }

        msg.textContent = text;
        chatContainer.appendChild(msg);
        msg.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    async function sendMessage() {
        const message = textarea.value.trim();
        if (!message) return;

        addMessage(message, "user");
        textarea.value = "";

        addMessage("Thinking...", "bot");
        const thinkingBubble = chatContainer.lastChild;

        try {
            const res = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message }),
            });

            const data = await res.json();

            thinkingBubble.textContent =
                data.reply || "No response from server.";
        } catch (err) {
            thinkingBubble.textContent =
                "Error connecting to backend.";
            console.error(err);
        }
    }

    // Button click
    sendButton.addEventListener("click", sendMessage);

    // Enter key submit
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});
// --------------------------------------------------