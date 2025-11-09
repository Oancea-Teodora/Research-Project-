package com.example.myapp1

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TradeAdapter(
    val items: MutableList<Trade>,
    private val onClick: (Trade) -> Unit
) : RecyclerView.Adapter<TradeAdapter.VH>() {

    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val line1: TextView = v.findViewById(R.id.tvLine1)
        val line2: TextView = v.findViewById(R.id.tvLine2)
    }

    private val fmt = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_trade, parent, false)
        return VH(v)
    }

    override fun getItemCount() = items.size

    override fun onBindViewHolder(h: VH, i: Int) {
        val t = items[i]
        h.line1.text = "${t.symbol} • ${t.type.uppercase()} • ${t.quantity} @ ${t.price}"
        h.line2.text = fmt.format(Date(t.date))
        h.itemView.setOnClickListener { onClick(t) }
    }
}
