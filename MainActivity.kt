package com.example.myapp1


import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class MainActivity : AppCompatActivity() {
    private lateinit var adapter: TradeAdapter
    private lateinit var count: TextView
    private lateinit var search: EditText

    private val openCreate = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val t = result.data?.getSerializableExtra("new_trade") as? Trade ?: return@registerForActivityResult
        TradeRepo.items.add(0, t)
        adapter.items.add(0, t)
        adapter.notifyItemInserted(0)
        updateCount()
    }

    private val openEdit = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val updated = result.data?.getSerializableExtra("updated_trade") as? Trade
        val deletedId = result.data?.getStringExtra("deleted_id")
        when {
            updated != null -> {
                val i = TradeRepo.indexOf(updated.id)
                if (i >= 0) {
                    TradeRepo.items[i] = updated
                    val j = adapter.items.indexOfFirst { it.id == updated.id }
                    if (j >= 0) { adapter.items[j] = updated; adapter.notifyItemChanged(j) }
                }
            }
            deletedId != null -> {
                val i = TradeRepo.indexOf(deletedId)
                if (i >= 0) TradeRepo.items.removeAt(i)
                val j = adapter.items.indexOfFirst { it.id == deletedId }
                if (j >= 0) { adapter.items.removeAt(j); adapter.notifyItemRemoved(j) }
                updateCount()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        count = findViewById(R.id.tvCount)
        search = findViewById(R.id.etSearch)
        val addBtn: Button = findViewById(R.id.btnAdd)
        val rv: RecyclerView = findViewById(R.id.rv)

        adapter = TradeAdapter(mutableListOf()) { trade ->
            val it = Intent(this, EditTradeActivity::class.java)
            it.putExtra("trade_id", trade.id)
            openEdit.launch(it)
        }
        rv.layoutManager = LinearLayoutManager(this)
        rv.adapter = adapter

        adapter.items.addAll(TradeRepo.items)
        adapter.notifyDataSetChanged()
        updateCount()

        addBtn.setOnClickListener {
            openCreate.launch(Intent(this, AddTradeActivity::class.java))
        }

        search.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {}
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                val q = s?.toString()?.trim()?.uppercase() ?: ""
                val list = if (q.isEmpty()) TradeRepo.items
                else TradeRepo.items.filter { it.symbol.uppercase().contains(q) }
                adapter.items.clear(); adapter.items.addAll(list); adapter.notifyDataSetChanged()
            }
        })
    }

    private fun updateCount() { count.text = "${TradeRepo.items.size} trades logged" }
}
