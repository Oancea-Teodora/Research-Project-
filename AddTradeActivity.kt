package com.example.myapp1

import android.app.DatePickerDialog
import android.content.Intent
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

class AddTradeActivity : AppCompatActivity() {
    private lateinit var etSymbol: EditText
    private lateinit var etPrice: EditText
    private lateinit var etQty: EditText
    private lateinit var etDate: EditText
    private lateinit var etNotes: EditText
    private lateinit var rbBuy: RadioButton
    private val fmt = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_add_edit)
        title = "Add Trade"

        etSymbol = findViewById(R.id.etSymbol)
        etPrice = findViewById(R.id.etPrice)
        etQty = findViewById(R.id.etQty)
        etDate = findViewById(R.id.etDate)
        etNotes = findViewById(R.id.etNotes)
        rbBuy = findViewById(R.id.rbBuy)
        val btnSave: Button = findViewById(R.id.btnSave)
        val btnDelete: Button = findViewById(R.id.btnDelete)
        btnDelete.isEnabled = false; btnDelete.alpha = 0.3f

        etDate.setText(fmt.format(Calendar.getInstance().time))
        etDate.setOnClickListener { pickDate() }

        btnSave.setOnClickListener {
            if (!valid()) return@setOnClickListener
            val t = Trade(
                symbol = etSymbol.text.toString().trim(),
                type = if (rbBuy.isChecked) "buy" else "sell",
                price = etPrice.text.toString().toDouble(),
                quantity = etQty.text.toString().toInt(),
                date = fmt.parse(etDate.text.toString())!!.time,
                notes = etNotes.text.toString()
            )
            setResult(RESULT_OK, Intent().putExtra("new_trade", t))
            finish()
        }
    }

    private fun pickDate() {
        val c = Calendar.getInstance()
        DatePickerDialog(this, { _, y, m, d ->
            c.set(y, m, d)
            etDate.setText(fmt.format(c.time))
        }, c.get(Calendar.YEAR), c.get(Calendar.MONTH), c.get(Calendar.DAY_OF_MONTH)).show()
    }

    private fun valid(): Boolean {
        var ok = true
        if (etSymbol.text.isBlank()) { etSymbol.error = "Required"; ok = false }
        if (etPrice.text.toString().toDoubleOrNull()?.let { it > 0 } != true) { etPrice.error = "> 0"; ok = false }
        if (etQty.text.toString().toIntOrNull()?.let { it > 0 } != true) { etQty.error = "> 0"; ok = false }
        if (etDate.text.isBlank()) { etDate.error = "Required"; ok = false }
        return ok
    }
}
