package com.example.myapp1

import android.app.AlertDialog
import android.app.DatePickerDialog
import android.content.Intent
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

class EditTradeActivity : AppCompatActivity() {
    private lateinit var trade: Trade
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
        title = "Edit Trade"

        val id = intent.getStringExtra("trade_id") ?: run { finish(); return }
        val idx = TradeRepo.indexOf(id)
        if (idx < 0) { finish(); return }
        trade = TradeRepo.items[idx]

        etSymbol = findViewById(R.id.etSymbol)
        etPrice = findViewById(R.id.etPrice)
        etQty = findViewById(R.id.etQty)
        etDate = findViewById(R.id.etDate)
        etNotes = findViewById(R.id.etNotes)
        rbBuy = findViewById(R.id.rbBuy)
        val rbSell: RadioButton = findViewById(R.id.rbSell)
        val btnSave: Button = findViewById(R.id.btnSave)
        val btnDelete: Button = findViewById(R.id.btnDelete)

        etSymbol.setText(trade.symbol)
        etPrice.setText(trade.price.toString())
        etQty.setText(trade.quantity.toString())
        etDate.setText(fmt.format(Date(trade.date)))
        etNotes.setText(trade.notes)
        if (trade.type == "buy") rbBuy.isChecked = true else rbSell.isChecked = true

        etDate.setOnClickListener { pickDate() }

        btnSave.setOnClickListener {
            if (!valid()) return@setOnClickListener
            trade.symbol = etSymbol.text.toString().trim()
            trade.type = if (rbBuy.isChecked) "buy" else "sell"
            trade.price = etPrice.text.toString().toDouble()
            trade.quantity = etQty.text.toString().toInt()
            trade.date = fmt.parse(etDate.text.toString())!!.time
            trade.notes = etNotes.text.toString()
            setResult(RESULT_OK, Intent().putExtra("updated_trade", trade))
            finish()
        }

        btnDelete.setOnClickListener {
            AlertDialog.Builder(this)
                .setMessage("Delete this trade?")
                .setPositiveButton("Delete") { _, _ ->
                    setResult(RESULT_OK, Intent().putExtra("deleted_id", trade.id))
                    finish()
                }
                .setNegativeButton("Cancel", null)
                .show()
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
