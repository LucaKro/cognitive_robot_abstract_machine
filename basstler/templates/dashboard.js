  function planDashboardCopyActionCommand(button) {
    var baseCommand = button.getAttribute("data-action-command");
    var modelPicker = button.previousElementSibling;
    var model = modelPicker && modelPicker.classList.contains("model-picker")
      ? modelPicker.getAttribute("data-model")
      : "";
    var command = model ? "/model " + model + "\n" + baseCommand : baseCommand;
    var originalLabel = button.textContent;
    var resetTimeoutMilliseconds = 1800;

    function showResult(className, label) {
      button.classList.remove("action-button-copied", "action-button-failed");
      button.classList.add(className);
      button.textContent = label;
      setTimeout(function () {
        button.classList.remove(className);
        button.textContent = originalLabel;
      }, resetTimeoutMilliseconds);
    }

    function onCopySucceeded() {
      showResult("action-button-copied", "Copied!");
    }

    function onCopyFailed() {
      showResult("action-button-failed", "Copy failed");
    }

    function copyViaTemporaryTextarea() {
      var textarea = document.createElement("textarea");
      textarea.value = command;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      var succeeded = false;
      try {
        succeeded = document.execCommand("copy");
      } catch (error) {
        succeeded = false;
      }
      document.body.removeChild(textarea);
      if (succeeded) {
        onCopySucceeded();
      } else {
        onCopyFailed();
      }
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(command).then(onCopySucceeded, copyViaTemporaryTextarea);
    } else {
      copyViaTemporaryTextarea();
    }
  }

  function planDashboardHighlightItem(event, link) {
    var identifier = link.getAttribute("data-item-identifier");
    var target = document.getElementById("item-" + identifier);
    var highlightMilliseconds = 2200;
    if (!target) {
      return;
    }
    event.preventDefault();
    target.scrollIntoView({behavior: "smooth", block: "center"});
    target.classList.remove("item-highlighted");
    void target.offsetWidth;
    target.classList.add("item-highlighted");
    window.setTimeout(function () {
      target.classList.remove("item-highlighted");
    }, highlightMilliseconds);
  }

  function planDashboardCloseModelPickers() {
    document.querySelectorAll(".model-picker.model-picker-open").forEach(function (picker) {
      picker.classList.remove("model-picker-open");
    });
  }

  function planDashboardToggleModelPicker(event, toggleButton) {
    event.stopPropagation();
    var picker = toggleButton.closest(".model-picker");
    var wasOpen = picker.classList.contains("model-picker-open");
    planDashboardCloseModelPickers();
    if (wasOpen) {
      return;
    }
    var list = picker.querySelector(".model-picker-list");
    var toggleRect = toggleButton.getBoundingClientRect();
    list.style.top = (toggleRect.bottom + 4) + "px";
    list.style.left = toggleRect.left + "px";
    picker.classList.add("model-picker-open");
  }

  function planDashboardSelectModel(event, optionElement) {
    event.stopPropagation();
    var picker = optionElement.closest(".model-picker");
    var toggleButton = picker.querySelector(".model-picker-toggle");
    picker.setAttribute("data-model", optionElement.getAttribute("data-value"));
    toggleButton.textContent = optionElement.textContent + " ▾";
    picker.querySelectorAll(".model-picker-option-selected").forEach(function (selected) {
      selected.classList.remove("model-picker-option-selected");
    });
    optionElement.classList.add("model-picker-option-selected");
    planDashboardCloseModelPickers();
  }

  document.addEventListener("click", planDashboardCloseModelPickers);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      planDashboardCloseModelPickers();
    }
  });
