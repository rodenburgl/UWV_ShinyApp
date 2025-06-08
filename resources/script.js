function update_button() {

    companysize = document.getElementById("Number_employees").value

    if (companysize >= 100) {
        document.getElementById("Confirm_premium_Button").disabled = true;
    } else {
        document.getElementById("Confirm_premium_Button").disabled = false;

    }
}

document.addEventListener("DOMContentLoaded", (event) => {
    document.getElementById("Number_employees").addEventListener("change", update_button)
});
